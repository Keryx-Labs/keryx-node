use keryx_addresses::Version;
use keryx_bip32::secp256k1;
use keryx_consensus_core::collateral::{escrow_delegation_message, verify_escrow_delegation};
use keryx_wallet_core::account::{BIP32_ACCOUNT_KIND, KEYPAIR_ACCOUNT_KIND};

use crate::imports::*;

/// Produces the H6 escrow delegation cert: a schnorr signature by a payout address key over an
/// escrow pubkey. The miner announces it as `/esig:<cert>` next to `/escrow:<pubkey>`; without a
/// valid pair, blocks are invalid past the gate.
#[derive(Default)]
pub struct DelegateEscrow;

#[async_trait]
impl Handler for DelegateEscrow {
    fn verb(&self, _ctx: &Arc<dyn Context>) -> Option<&'static str> {
        Some("delegate-escrow")
    }

    fn help(&self, _ctx: &Arc<dyn Context>) -> &'static str {
        "Sign an escrow delegation cert (H6) binding an escrow key to a payout address"
    }

    async fn handle(self: Arc<Self>, ctx: &Arc<dyn Context>, argv: Vec<String>, cmd: &str) -> cli::Result<()> {
        let ctx = ctx.clone().downcast_arc::<KaspaCli>()?;
        self.main(ctx, argv, cmd).await.map_err(|e| e.into())
    }
}

impl DelegateEscrow {
    async fn main(self: Arc<Self>, ctx: Arc<KaspaCli>, argv: Vec<String>, _cmd: &str) -> Result<()> {
        if argv.is_empty() || argv.len() > 2 {
            return self.display_help(ctx).await;
        }

        let escrow_hex = argv[0].as_str();
        let mut escrow_pubkey = [0u8; 32];
        if escrow_hex.len() != 64 || faster_hex::hex_decode(escrow_hex.as_bytes(), &mut escrow_pubkey).is_err() {
            return Err(Error::custom("escrow pubkey must be 64 hex characters (the miner prints it at startup)"));
        }

        let address = match argv.get(1) {
            Some(addr) => Address::try_from(addr.as_str())?,
            None => {
                let account = ctx.wallet().account()?;
                account.receive_address()?
            }
        };
        if address.version != Version::PubKey {
            return Err(Error::custom("only schnorr PubKey payout addresses can delegate an escrow key"));
        }

        let privkey = self.get_address_private_key(&ctx, &address).await?;
        let keypair = secp256k1::Keypair::from_seckey_slice(secp256k1::SECP256K1, &privkey)
            .map_err(|_| Error::custom("invalid private key"))?;
        let msg = secp256k1::Message::from_digest(escrow_delegation_message(&escrow_pubkey));
        let sig = keypair.sign_schnorr(msg);
        let cert: [u8; 64] = *sig.as_ref();

        // Self-check against the exact consensus rule before handing the cert out.
        let payout_spk = keryx_txscript::pay_to_address_script(&address);
        if !verify_escrow_delegation(payout_spk.version(), payout_spk.script(), &escrow_pubkey, &cert) {
            return Err(Error::custom("cert self-check failed: the address key does not match the signing key"));
        }

        let cert_hex = faster_hex::hex_string(&cert);
        tprintln!(ctx, "Payout address : {}", address);
        tprintln!(ctx, "Escrow pubkey  : {}", escrow_hex);
        tprintln!(ctx, "Delegation cert: {}", cert_hex);
        tprintln!(ctx, "");
        tprintln!(ctx, "Save the cert to a file and pass it to the miner with --escrow-cert-file.");
        tprintln!(ctx, "Mine with this exact payout address: the cert only verifies against it.");

        Ok(())
    }

    async fn display_help(self: Arc<Self>, ctx: Arc<KaspaCli>) -> Result<()> {
        ctx.term().help(
            &[(
                "delegate-escrow <escrow_pubkey_hex> [payout_address]",
                "Sign the delegation cert binding the escrow key to the payout address (defaults to the account receive address)",
            )],
            None,
        )?;

        Ok(())
    }

    async fn get_address_private_key(self: Arc<Self>, ctx: &Arc<KaspaCli>, address: &Address) -> Result<[u8; 32]> {
        let account = ctx.wallet().account()?;

        match account.account_kind().as_ref() {
            BIP32_ACCOUNT_KIND => {
                let (wallet_secret, payment_secret) = ctx.ask_wallet_secret(Some(&account)).await?;
                let keydata = account.prv_key_data(wallet_secret).await?;
                let account = account.clone().as_derivation_capable().expect("expecting derivation capable");

                let (receive, change) = account.derivation().addresses_indexes(&[address])?;
                let private_keys = account.create_private_keys(&keydata, &payment_secret, &receive, &change)?;
                for (candidate, private_key) in private_keys {
                    if *address == *candidate {
                        return Ok(private_key.secret_bytes());
                    }
                }

                Err(Error::custom("Could not find address in any derivation path in account"))
            }
            KEYPAIR_ACCOUNT_KIND => {
                let (wallet_secret, payment_secret) = ctx.ask_wallet_secret(Some(&account)).await?;
                let keydata = account.prv_key_data(wallet_secret).await?;
                let decrypted_privkey = keydata.payload.decrypt(payment_secret.as_ref()).unwrap();
                let secretkey = decrypted_privkey.as_secret_key()?.unwrap();
                Ok(secretkey.secret_bytes())
            }
            _ => Err(Error::custom("Unsupported account kind")),
        }
    }
}
