use crate::imports::*;
use keryx_consensus_core::config::params::{INFERENCE_REWARD_MINIMUMS_V2_H6, POM_TIERS_H6};
use keryx_consensus_core::constants::{SOMPI_PER_KASPA, TX_VERSION};
use keryx_consensus_core::sign::{Signed, sign_with_multiple_v2};
use keryx_consensus_core::subnets::{SUBNETWORK_ID_AI_REQUEST, SUBNETWORK_ID_AI_RESPONSE};
use keryx_consensus_core::tx::{
    MutableTransaction, ScriptPublicKey, Transaction, TransactionInput, TransactionOutpoint, TransactionOutput, UtxoEntry,
};
use keryx_inference::{
    AiResponsePayload, INFERENCE_REWARD_TOKEN_STEP, INFERENCE_VAULT_SCRIPT, MAX_PRIVATE_RECIPIENTS, MIN_AI_REQUEST_PRIORITY_FEE,
    PrivateRequestSecret, open_response, seal_request,
};
use keryx_rpc_core::{RpcHash, RpcTransaction};
use keryx_txscript::pay_to_address_script;
use keryx_wallet_core::account::{BIP32_ACCOUNT_KIND, KEYPAIR_ACCOUNT_KIND};
use std::str::FromStr;
use std::time::{Duration, Instant};
use workflow_core::task::sleep;

/// Display names of the H6 tier lineup (index = `Header.pom_tier`), see `POM_TIERS_H6`.
const TIER_NAMES: [&str; 5] = ["Qwen3.5-9B", "GLM-4-9B", "Gemma-4-12B", "Qwen3.6-27B", "Kimi-Linear-48B"];
/// A request always leaves at least this much change: a tiny change output makes the KIP-9
/// storage mass of the transaction exceed the standard limit and the mempool rejects it.
const MIN_CHANGE_SOMPI: u64 = SOMPI_PER_KASPA;
const MAX_INPUTS: usize = 32;
const DEFAULT_MAX_TOKENS: u32 = 256;
const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// Private AI inference from the wallet: pick responders, seal a prompt to them, submit the
/// `AiRequest`, then wait for the sealed answer and decrypt it. The prompt and the answer are
/// readable only by this wallet and the named responders; everything else on the chain is
/// public. See `docs/private-inference.md`.
#[derive(Default, Handler)]
#[help("Private AI inference: list responders, send a sealed request, fetch and decrypt the answer")]
pub struct Inference;

impl Inference {
    async fn main(self: Arc<Self>, ctx: &Arc<dyn Context>, argv: Vec<String>, _cmd: &str) -> Result<()> {
        let ctx = ctx.clone().downcast_arc::<KaspaCli>()?;
        let Some(sub) = argv.first() else {
            return self.display_help(ctx).await;
        };
        match sub.as_str() {
            "providers" => self.providers(&ctx, &argv[1..]).await,
            "send" => self.send(&ctx, &argv[1..]).await,
            "fetch" => self.fetch(&ctx, &argv[1..]).await,
            "decrypt" => self.decrypt(&ctx, &argv[1..]).await,
            _ => self.display_help(ctx).await,
        }
    }

    async fn display_help(self: Arc<Self>, ctx: Arc<KaspaCli>) -> Result<()> {
        ctx.term().help(
            &[
                ("inference providers [tier]", "List the responders eligible right now per tier, with the escrow key to seal requests to"),
                (
                    "inference send --to <escrow>[,<escrow>..] --tier <0-4> [--max-tokens N] [--reward KRX] [--fee KRX] [--wait SECS] <prompt | @file>",
                    "Seal a prompt to the named responders (up to 16) and submit the AiRequest",
                ),
                (
                    "inference fetch <request id> <root key> [--since <block hash>] [--timeout SECS]",
                    "Wait for the sealed answer of a request and decrypt it",
                ),
                (
                    "inference decrypt <request id> <root key> <responder escrow> <@file | hex>",
                    "Decrypt an answer body fetched from IPFS (a response without an inline body)",
                ),
            ],
            None,
        )?;
        Ok(())
    }

    // ── providers ───────────────────────────────────────────────────────────────────────────

    async fn providers(&self, ctx: &Arc<KaspaCli>, args: &[String]) -> Result<()> {
        let tier_filter = match args.first() {
            Some(t) => Some(t.parse::<u32>().map_err(|_| Error::custom("tier must be a number 0-4"))?),
            None => None,
        };
        let resp = ctx.wallet().rpc_api().get_service_providers().await?;
        tprintln!(ctx, "Service-eligible responders at DAA {} (tier, model, escrow key, identity):", resp.virtual_daa_score);
        let mut shown = 0;
        for p in resp.providers.iter().filter(|p| tier_filter.is_none_or(|t| t == p.tier)) {
            let name = TIER_NAMES.get(p.tier as usize).copied().unwrap_or("?");
            tprintln!(ctx, "  tier {} {:<16} escrow {}  identity {}", p.tier, name, p.escrow_pubkey, p.identity);
            shown += 1;
        }
        if shown == 0 {
            tprintln!(ctx, "  (none: no miner proved a block of that tier inside the eligibility window)");
        } else {
            tprintln!(ctx, "");
            tprintln!(ctx, "A private request only obligates and pays the responders it names; name a listed escrow key,");
            tprintln!(ctx, "or several for redundancy, with `inference send --to <escrow>[,<escrow>..]`.");
        }
        Ok(())
    }

    // ── send ────────────────────────────────────────────────────────────────────────────────

    async fn send(&self, ctx: &Arc<KaspaCli>, args: &[String]) -> Result<()> {
        let mut recipients: Vec<[u8; 32]> = Vec::new();
        let mut tier: Option<usize> = None;
        let mut model: Option<[u8; 32]> = None;
        let mut max_tokens = DEFAULT_MAX_TOKENS;
        let mut reward: Option<u64> = None;
        let mut fee = MIN_AI_REQUEST_PRIORITY_FEE;
        let mut wait: Option<u64> = None;
        let mut prompt_words: Vec<String> = Vec::new();

        let mut i = 0;
        while i < args.len() {
            let arg = args[i].as_str();
            let value = |i: usize| -> Result<&String> { args.get(i + 1).ok_or_else(|| Error::custom(format!("{arg} needs a value"))) };
            match arg {
                "--to" => {
                    for key in value(i)?.split(',').filter(|s| !s.is_empty()) {
                        recipients.push(parse_hex32(key, "escrow key")?);
                    }
                    i += 2;
                }
                "--tier" => {
                    tier = Some(value(i)?.parse::<usize>().map_err(|_| Error::custom("--tier must be a number 0-4"))?);
                    i += 2;
                }
                "--model" => {
                    model = Some(parse_hex32(value(i)?, "model id")?);
                    i += 2;
                }
                "--max-tokens" => {
                    max_tokens = value(i)?.parse::<u32>().map_err(|_| Error::custom("--max-tokens must be a number"))?;
                    i += 2;
                }
                "--reward" => {
                    reward = Some(try_parse_required_nonzero_kaspa_as_sompi_u64(Some(value(i)?))?);
                    i += 2;
                }
                "--fee" => {
                    fee = try_parse_required_nonzero_kaspa_as_sompi_u64(Some(value(i)?))?;
                    i += 2;
                }
                "--wait" => {
                    wait = Some(value(i)?.parse::<u64>().map_err(|_| Error::custom("--wait must be seconds"))?);
                    i += 2;
                }
                _ => {
                    prompt_words.push(args[i].clone());
                    i += 1;
                }
            }
        }

        if recipients.is_empty() {
            return Err(Error::custom("name at least one responder with --to <escrow key> (see `inference providers`)"));
        }
        if recipients.len() > MAX_PRIVATE_RECIPIENTS {
            return Err(Error::custom(format!("at most {MAX_PRIVATE_RECIPIENTS} responders can be named")));
        }
        let model_id = match (model, tier) {
            (Some(m), _) => m,
            (None, Some(t)) => POM_TIERS_H6.get(t).map(|m| m.model_id).ok_or_else(|| Error::custom("--tier must be 0-4"))?,
            (None, None) => return Err(Error::custom("choose the model with --tier <0-4> or --model <hex>")),
        };
        let prompt = read_text_argument(&prompt_words)?;
        if prompt.is_empty() {
            return Err(Error::custom("the prompt is empty: pass it as words after the options, or as @<file>"));
        }
        let reward = match reward {
            Some(r) => r,
            None => {
                let base = INFERENCE_REWARD_MINIMUMS_V2_H6
                    .iter()
                    .find(|(id, _)| *id == model_id)
                    .map(|(_, base)| *base)
                    .ok_or_else(|| Error::custom("this model has no known reward floor: pass --reward <KRX>"))?;
                base + (max_tokens as u64).div_ceil(64) * INFERENCE_REWARD_TOKEN_STEP
            }
        };
        if fee < MIN_AI_REQUEST_PRIORITY_FEE {
            return Err(Error::custom(format!(
                "--fee is below the network minimum of {} KRX",
                sompi_to_kaspa_string(MIN_AI_REQUEST_PRIORITY_FEE)
            )));
        }

        // Seal the prompt: only the named escrow keys can open it, and the root key stays here.
        let (payload, secret) = seal_request(model_id, max_tokens, reward, fee, prompt.as_bytes(), &recipients)
            .map_err(|e| Error::custom(e.to_string()))?;

        // Fund it from the wallet: inputs covering reward + fee + change, change at outputs[0],
        // the keyless reward vault at outputs[1] (the layout consensus checks).
        let account = ctx.wallet().account()?;
        let needed = reward + fee + MIN_CHANGE_SOMPI;
        let mut utxos = account.utxo_context().get_utxos(None, None).await?;
        utxos.sort_by_key(|u| std::cmp::Reverse(u.amount));
        let mut selected = Vec::new();
        let mut total = 0u64;
        for utxo in utxos {
            if total >= needed || selected.len() == MAX_INPUTS {
                break;
            }
            total += utxo.amount;
            selected.push(utxo);
        }
        if total < needed {
            return Err(Error::custom(format!(
                "insufficient mature funds: the request needs {} KRX (reward {} + fee {} + at least {} change)",
                sompi_to_kaspa_string(needed),
                sompi_to_kaspa_string(reward),
                sompi_to_kaspa_string(fee),
                sompi_to_kaspa_string(MIN_CHANGE_SOMPI)
            )));
        }
        let change = total - reward - fee;
        let change_address = account.change_address()?;
        let mut addresses: Vec<Address> = selected.iter().filter_map(|u| u.address.clone()).collect();
        addresses.sort();
        addresses.dedup();

        let inputs: Vec<TransactionInput> =
            selected.iter().map(|u| TransactionInput::new(TransactionOutpoint::from(&u.outpoint), vec![], 0, 1)).collect();
        let outputs = vec![
            TransactionOutput::new(change, pay_to_address_script(&change_address)),
            TransactionOutput::new(reward, ScriptPublicKey::new(0, INFERENCE_VAULT_SCRIPT.to_vec().into())),
        ];
        let entries: Vec<UtxoEntry> =
            selected.iter().map(|u| UtxoEntry::new(u.amount, u.script_public_key.clone(), u.block_daa_score, u.is_coinbase)).collect();
        let tx = Transaction::new_non_finalized(TX_VERSION, inputs, outputs, 0, SUBNETWORK_ID_AI_REQUEST, 0, payload.serialize());

        let keys = self.private_keys_for(ctx, &account, &addresses).await?;
        let mut tx = match sign_with_multiple_v2(MutableTransaction::with_entries(tx, entries), &keys) {
            Signed::Fully(mtx) => mtx.tx,
            Signed::Partially(_) => return Err(Error::custom("could not sign every input with the account keys")),
        };
        tx.finalize();
        let request_id = tx.id();

        let rpc = ctx.wallet().rpc_api();
        // The answer can only appear after the request: remember where the DAG was.
        let since = rpc.get_block_dag_info().await?.sink;
        let rpc_tx: RpcTransaction = (&tx).into();
        rpc.submit_transaction(rpc_tx, false).await?;

        tprintln!(ctx, "Private inference request submitted");
        tprintln!(ctx, "  request id : {request_id}");
        tprintln!(ctx, "  root key   : {}   (keep it: the answer cannot be read without it)", secret.to_hex());
        tprintln!(
            ctx,
            "  model      : {}  max_tokens {max_tokens}  reward {} KRX  fee {} KRX",
            hex::encode(model_id),
            sompi_to_kaspa_string(reward),
            sompi_to_kaspa_string(fee)
        );
        tprintln!(ctx, "  responders : {}", recipients.iter().map(hex::encode).collect::<Vec<_>>().join(", "));
        tprintln!(ctx, "  since block: {since}");
        tprintln!(ctx, "");
        tprintln!(ctx, "Fetch the answer later with: inference fetch {request_id} {} --since {since}", secret.to_hex());

        if let Some(timeout) = wait {
            tprintln!(ctx, "");
            self.wait_for_answer(ctx, request_id.as_bytes(), &secret, since, timeout).await?;
        }
        Ok(())
    }

    /// The spending keys of `addresses` in the account: BIP32 accounts derive them per address,
    /// keypair accounts hold a single one.
    async fn private_keys_for(&self, ctx: &Arc<KaspaCli>, account: &Arc<dyn Account>, addresses: &[Address]) -> Result<Vec<[u8; 32]>> {
        let (wallet_secret, payment_secret) = ctx.ask_wallet_secret(Some(account)).await?;
        let keydata = account.prv_key_data(wallet_secret).await?;
        match account.account_kind().as_ref() {
            BIP32_ACCOUNT_KIND => {
                let account = account.clone().as_derivation_capable()?;
                let refs: Vec<&Address> = addresses.iter().collect();
                let (receive, change) = account.derivation().addresses_indexes(&refs)?;
                let keys = account.create_private_keys(&keydata, &payment_secret, &receive, &change)?;
                if keys.len() < addresses.len() {
                    return Err(Error::custom("could not derive a key for every input address of this account"));
                }
                Ok(keys.into_iter().map(|(_, key)| key.secret_bytes()).collect())
            }
            KEYPAIR_ACCOUNT_KIND => {
                let decrypted = keydata.payload.decrypt(payment_secret.as_ref())?;
                let key = decrypted.as_secret_key()?.ok_or_else(|| Error::custom("the keypair account has no secret key"))?;
                Ok(vec![key.secret_bytes()])
            }
            _ => Err(Error::custom("unsupported account kind")),
        }
    }

    // ── fetch ───────────────────────────────────────────────────────────────────────────────

    async fn fetch(&self, ctx: &Arc<KaspaCli>, args: &[String]) -> Result<()> {
        if args.len() < 2 {
            return Err(Error::custom("usage: inference fetch <request id> <root key> [--since <block hash>] [--timeout SECS]"));
        }
        let request_hash = parse_hex32(&args[0], "request id")?;
        let secret = PrivateRequestSecret::from_hex(&args[1]).map_err(|e| Error::custom(e.to_string()))?;
        let mut since: Option<RpcHash> = None;
        let mut timeout = DEFAULT_TIMEOUT_SECS;
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--since" => {
                    let v = args.get(i + 1).ok_or_else(|| Error::custom("--since needs a block hash"))?;
                    since = Some(RpcHash::from_str(v).map_err(|_| Error::custom("--since must be a block hash"))?);
                    i += 2;
                }
                "--timeout" => {
                    let v = args.get(i + 1).ok_or_else(|| Error::custom("--timeout needs seconds"))?;
                    timeout = v.parse::<u64>().map_err(|_| Error::custom("--timeout must be seconds"))?;
                    i += 2;
                }
                other => return Err(Error::custom(format!("unknown option {other}"))),
            }
        }
        let since = match since {
            Some(h) => h,
            None => ctx.wallet().rpc_api().get_block_dag_info().await?.sink,
        };
        self.wait_for_answer(ctx, request_hash, &secret, since, timeout).await
    }

    /// Polls the mempool and every block past `since` for a signed `AiResponse` to
    /// `request_hash`, and decrypts the first inline body that opens with `secret`.
    async fn wait_for_answer(
        &self,
        ctx: &Arc<KaspaCli>,
        request_hash: [u8; 32],
        secret: &PrivateRequestSecret,
        since: RpcHash,
        timeout: u64,
    ) -> Result<()> {
        let rpc = ctx.wallet().rpc_api();
        let deadline = Instant::now() + Duration::from_secs(timeout);
        let mut low = since;
        let mut reported: Vec<RpcHash> = Vec::new();
        tprintln!(ctx, "Waiting up to {timeout}s for a sealed answer to {}...", hex::encode(request_hash));
        loop {
            let mut candidates: Vec<RpcTransaction> = Vec::new();
            for entry in rpc.get_mempool_entries(false, false).await? {
                candidates.push(entry.transaction);
            }
            let blocks = rpc.get_blocks(Some(low), true, true).await?;
            for block in blocks.blocks {
                candidates.extend(block.transactions);
            }
            if let Some(last) = blocks.block_hashes.last() {
                low = *last;
            }
            for tx in candidates {
                if tx.subnetwork_id != SUBNETWORK_ID_AI_RESPONSE {
                    continue;
                }
                let Some(resp) = AiResponsePayload::deserialize(&tx.payload) else { continue };
                if resp.request_hash != request_hash {
                    continue;
                }
                let txid = tx.verbose_data.as_ref().map(|v| v.transaction_id).unwrap_or_default();
                if reported.contains(&txid) {
                    continue;
                }
                reported.push(txid);
                let Some(responder) = resp.responder.as_ref() else {
                    tprintln!(ctx, "Ignoring an unsigned (v1) response: it cannot carry a sealed answer.");
                    continue;
                };
                match resp.private_body.as_deref() {
                    Some(body) => match open_response(&secret.root_key, &request_hash, &responder.escrow_pubkey, body) {
                        Ok(answer) => {
                            tprintln!(
                                ctx,
                                "Answer from responder {} ({} tokens):",
                                hex::encode(responder.escrow_pubkey),
                                resp.response_length
                            );
                            tprintln!(ctx, "");
                            tprintln!(ctx, "{}", String::from_utf8_lossy(&answer));
                            return Ok(());
                        }
                        Err(e) => tprintln!(
                            ctx,
                            "A response from {} did not open ({e}); still waiting.",
                            hex::encode(responder.escrow_pubkey)
                        ),
                    },
                    None => {
                        tprintln!(
                            ctx,
                            "Responder {} published its answer on IPFS only: CID {}",
                            hex::encode(responder.escrow_pubkey),
                            resp.cid_v0()
                        );
                        tprintln!(
                            ctx,
                            "Fetch the CID and decrypt it with: inference decrypt {} {} {} @<file>",
                            hex::encode(request_hash),
                            secret.to_hex(),
                            hex::encode(responder.escrow_pubkey)
                        );
                        return Ok(());
                    }
                }
            }
            if Instant::now() >= deadline {
                tprintln!(
                    ctx,
                    "No answer within {timeout}s. Retry with: inference fetch {} {} --since {low}",
                    hex::encode(request_hash),
                    secret.to_hex()
                );
                return Ok(());
            }
            sleep(Duration::from_secs(2)).await;
        }
    }

    // ── decrypt ─────────────────────────────────────────────────────────────────────────────

    async fn decrypt(&self, ctx: &Arc<KaspaCli>, args: &[String]) -> Result<()> {
        if args.len() < 4 {
            return Err(Error::custom("usage: inference decrypt <request id> <root key> <responder escrow> <@file | hex>"));
        }
        let request_hash = parse_hex32(&args[0], "request id")?;
        let secret = PrivateRequestSecret::from_hex(&args[1]).map_err(|e| Error::custom(e.to_string()))?;
        let responder = parse_hex32(&args[2], "responder escrow key")?;
        let body = match args[3].strip_prefix('@') {
            Some(path) => std::fs::read(path).map_err(|e| Error::custom(format!("cannot read {path}: {e}")))?,
            None => hex::decode(args[3].trim()).map_err(|_| Error::custom("the answer body must be @<file> or hex"))?,
        };
        let answer = open_response(&secret.root_key, &request_hash, &responder, &body).map_err(|e| Error::custom(e.to_string()))?;
        tprintln!(ctx, "{}", String::from_utf8_lossy(&answer));
        Ok(())
    }
}

fn parse_hex32(s: &str, what: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(s.trim()).map_err(|_| Error::custom(format!("{what} must be 64 hex characters")))?;
    bytes.try_into().map_err(|_| Error::custom(format!("{what} must be 32 bytes (64 hex characters)")))
}

/// The prompt: the words after the options joined by spaces, or the exact content of `@<file>`.
fn read_text_argument(words: &[String]) -> Result<String> {
    if let [single] = words
        && let Some(path) = single.strip_prefix('@')
    {
        return std::fs::read_to_string(path).map_err(|e| Error::custom(format!("cannot read {path}: {e}")));
    }
    Ok(words.join(" "))
}
