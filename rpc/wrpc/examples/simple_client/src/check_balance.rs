/// Query the balance and UTXOs of a Keryx address via wRPC Borsh.
///
/// Usage:
///   cargo run -p keryx-wrpc-simple-client-example --bin check-balance --release -- \
///     keryx:qzcw2l22ge2sjch4fv9rxtt8eghyszswpe5jxtkwk53phgw4a8z35gtagxpx4
///
///   Or against a remote node:
///   ... -- <address> ws://1.2.3.4:23110
use std::process::ExitCode;
use std::time::Duration;

use keryx_rpc_core::api::rpc::RpcApi;
use keryx_wrpc_client::{
    KaspaRpcClient, WrpcEncoding,
    client::{ConnectOptions, ConnectStrategy},
    prelude::{NetworkId, NetworkType},
    result::Result,
};
use keryx_rpc_core::RpcAddress as Address;

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: check-balance <keryx-address> [ws://host:port]");
        eprintln!("Default node: ws://127.0.0.1:23110");
        return ExitCode::FAILURE;
    }

    let address_str = &args[1];
    let node_url = args.get(2).map(|s| s.as_str()).unwrap_or("ws://127.0.0.1:23110");

    match query(address_str, node_url).await {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn query(address_str: &str, node_url: &str) -> Result<()> {
    let client = KaspaRpcClient::new(
        WrpcEncoding::Borsh,
        Some(node_url),
        None,
        Some(NetworkId::new(NetworkType::Mainnet)),
        None,
    )?;

    let options = ConnectOptions {
        block_async_connect: true,
        connect_timeout: Some(Duration::from_millis(5_000)),
        strategy: ConnectStrategy::Fallback,
        ..Default::default()
    };

    client.connect(Some(options)).await?;

    let address: Address = address_str.try_into()?;

    // Balance
    let balance = client.get_balance_by_address(address.clone()).await?;
    let krx = balance as f64 / 1e8;
    println!("Address : {}", address_str);
    println!("Balance : {} sompi ({:.8} KRX)", balance, krx);

    // UTXOs detail
    let utxos = client.get_utxos_by_addresses(vec![address]).await?;
    if utxos.is_empty() {
        println!("UTXOs   : none");
    } else {
        println!("UTXOs   : {} entries", utxos.len());
        for (i, entry) in utxos.iter().enumerate() {
            println!(
                "  [{}] txid={} index={} amount={} sompi (confirmed block DAA: {})",
                i,
                entry.outpoint.transaction_id,
                entry.outpoint.index,
                entry.utxo_entry.amount,
                entry.utxo_entry.block_daa_score,
            );
        }
    }

    client.disconnect().await?;
    Ok(())
}
