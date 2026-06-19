//! Conversion of Block related types

use std::sync::Arc;

use crate::{RpcBlock, RpcError, RpcOptionalBlock, RpcOptionalTransaction, RpcRawBlock, RpcResult, RpcTransaction};
use keryx_consensus_core::block::{Block, MutableBlock};
use keryx_consensus_core::pom::PomProof;

// ----------------------------------------------------------------------------
// consensus_core to rpc_core
// ----------------------------------------------------------------------------

impl From<&Block> for RpcBlock {
    fn from(item: &Block) -> Self {
        Self {
            header: item.header.as_ref().into(),
            transactions: item.transactions.iter().map(RpcTransaction::from).collect(),
            // TODO: Implement a populating process inspired from keryxd\app\rpc\rpccontext\verbosedata.go
            verbose_data: None,
        }
    }
}

impl From<&Block> for RpcRawBlock {
    fn from(item: &Block) -> Self {
        Self {
            header: item.header.as_ref().into(),
            transactions: item.transactions.iter().map(RpcTransaction::from).collect(),
            pom_proof: item.pom_proof.as_ref().map(|p| borsh::to_vec(p.as_ref()).expect("PomProof borsh serialize")),
        }
    }
}

impl From<&MutableBlock> for RpcBlock {
    fn from(item: &MutableBlock) -> Self {
        Self {
            header: item.header.as_ref().into(),
            transactions: item.transactions.iter().map(RpcTransaction::from).collect(),
            verbose_data: None,
        }
    }
}

impl From<&MutableBlock> for RpcRawBlock {
    fn from(item: &MutableBlock) -> Self {
        Self {
            header: item.header.as_ref().into(),
            transactions: item.transactions.iter().map(RpcTransaction::from).collect(),
            pom_proof: None,
        }
    }
}

impl From<MutableBlock> for RpcRawBlock {
    fn from(item: MutableBlock) -> Self {
        Self {
            header: item.header.into(),
            transactions: item.transactions.iter().map(RpcTransaction::from).collect(),
            pom_proof: None,
        }
    }
}

// ----------------------------------------------------------------------------
// rpc_core to consensus_core
// ----------------------------------------------------------------------------

impl TryFrom<RpcBlock> for Block {
    type Error = RpcError;
    fn try_from(item: RpcBlock) -> RpcResult<Self> {
        Ok(Self {
            header: Arc::new(item.header.try_into()?),
            transactions: Arc::new(
                item.transactions
                    .into_iter()
                    .map(keryx_consensus_core::tx::Transaction::try_from)
                    .collect::<RpcResult<Vec<keryx_consensus_core::tx::Transaction>>>()?,
            ),
            // RpcBlock (verbose) does not carry the PoM proof; submit uses RpcRawBlock.
            pom_proof: None,
        })
    }
}

impl TryFrom<RpcRawBlock> for Block {
    type Error = RpcError;
    fn try_from(item: RpcRawBlock) -> RpcResult<Self> {
        let pom_proof = match &item.pom_proof {
            Some(bytes) => Some(Arc::new(
                borsh::from_slice::<PomProof>(bytes).map_err(|e| RpcError::PomProofDecodeError(e.to_string()))?,
            )),
            None => None,
        };
        Ok(Self {
            header: Arc::new(item.header.try_into()?),
            transactions: Arc::new(
                item.transactions
                    .into_iter()
                    .map(keryx_consensus_core::tx::Transaction::try_from)
                    .collect::<RpcResult<Vec<keryx_consensus_core::tx::Transaction>>>()?,
            ),
            pom_proof,
        })
    }
}

// ----------------------------------------------------------------------------
// consensus_core to optional rpc_core
// ----------------------------------------------------------------------------

impl From<&Block> for RpcOptionalBlock {
    fn from(item: &Block) -> Self {
        Self {
            header: Some(item.header.as_ref().into()),
            transactions: item.transactions.iter().map(RpcOptionalTransaction::from).collect(),
            // TODO: Implement a populating process inspired from keryxd\app\rpc\rpccontext\verbosedata.go
            verbose_data: None,
        }
    }
}

impl From<&MutableBlock> for RpcOptionalBlock {
    fn from(item: &MutableBlock) -> Self {
        Self {
            header: Some(item.header.as_ref().into()),
            transactions: item.transactions.iter().map(RpcOptionalTransaction::from).collect(),
            verbose_data: None,
        }
    }
}

// ----------------------------------------------------------------------------
// optional rpc_core to consensus_core
// ----------------------------------------------------------------------------

impl TryFrom<RpcOptionalBlock> for Block {
    type Error = RpcError;
    fn try_from(item: RpcOptionalBlock) -> RpcResult<Self> {
        Ok(Self {
            header: Arc::new(
                (item.header.ok_or(RpcError::MissingRpcFieldError("RpcBlock".to_string(), "header".to_string()))?).try_into()?,
            ),
            transactions: Arc::new(
                item.transactions
                    .into_iter()
                    .map(keryx_consensus_core::tx::Transaction::try_from)
                    .collect::<RpcResult<Vec<keryx_consensus_core::tx::Transaction>>>()?,
            ),
            // RpcOptionalBlock has no PoM proof field.
            pom_proof: None,
        })
    }
}
