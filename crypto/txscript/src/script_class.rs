use crate::{MAX_SCRIPT_PUBLIC_KEY_VERSION, opcodes};
use borsh::{BorshDeserialize, BorshSerialize};
use keryx_addresses::Version;
use keryx_consensus_core::tx::{ScriptPublicKey, ScriptPublicKeyVersion};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Display, Formatter},
    str::FromStr,
};
use thiserror::Error;

#[derive(Error, PartialEq, Eq, Debug, Clone)]
pub enum Error {
    #[error("Invalid script class {0}")]
    InvalidScriptClass(String),
}

/// Standard classes of script payment in the blockDAG
#[derive(PartialEq, Eq, Hash, Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
#[borsh(use_discriminant = true)]
#[repr(u8)]
pub enum ScriptClass {
    /// None of the recognized forms
    NonStandard = 0,
    /// Pay to pubkey
    PubKey,
    /// Pay to pubkey ECDSA
    PubKeyECDSA,
    /// Pay to script hash
    ScriptHash,
    /// CSV-timelocked pay to pubkey (OPoI escrow outputs)
    CsvPubKey = 4,
}

const NON_STANDARD: &str = "nonstandard";
const PUB_KEY: &str = "pubkey";
const PUB_KEY_ECDSA: &str = "pubkeyecdsa";
const SCRIPT_HASH: &str = "scripthash";
const CSV_PUB_KEY: &str = "csvpubkey";

impl ScriptClass {
    pub fn from_script(script_public_key: &ScriptPublicKey) -> Self {
        let script_public_key_ = script_public_key.script();
        if script_public_key.version() == MAX_SCRIPT_PUBLIC_KEY_VERSION {
            if Self::is_pay_to_pubkey(script_public_key_) {
                ScriptClass::PubKey
            } else if Self::is_pay_to_pubkey_ecdsa(script_public_key_) {
                Self::PubKeyECDSA
            } else if Self::is_pay_to_script_hash(script_public_key_) {
                Self::ScriptHash
            } else if Self::is_csv_pay_to_pubkey(script_public_key_) {
                Self::CsvPubKey
            } else {
                ScriptClass::NonStandard
            }
        } else {
            ScriptClass::NonStandard
        }
    }

    // Returns true if the script passed is a pay-to-pubkey
    // transaction, false otherwise.
    #[inline(always)]
    pub fn is_pay_to_pubkey(script_public_key: &[u8]) -> bool {
        (script_public_key.len() == 34) && // 2 opcodes number + 32 data
        (script_public_key[0] == opcodes::codes::OpData32) &&
        (script_public_key[33] == opcodes::codes::OpCheckSig)
    }

    // Returns returns true if the script passed is an ECDSA pay-to-pubkey
    /// transaction, false otherwise.
    #[inline(always)]
    pub fn is_pay_to_pubkey_ecdsa(script_public_key: &[u8]) -> bool {
        (script_public_key.len() == 35) && // 2 opcodes number + 33 data
        (script_public_key[0] == opcodes::codes::OpData33) &&
        (script_public_key[34] == opcodes::codes::OpCheckSigECDSA)
    }

    /// Returns true if the script is in the standard
    /// pay-to-script-hash (P2SH) format, false otherwise.
    #[inline(always)]
    pub fn is_pay_to_script_hash(script_public_key: &[u8]) -> bool {
        (script_public_key.len() == 35) && // 3 opcodes number + 32 data
        (script_public_key[0] == opcodes::codes::OpBlake2b) &&
        (script_public_key[1] == opcodes::codes::OpData32) &&
        (script_public_key[34] == opcodes::codes::OpEqual)
    }

    /// Returns true if the script is a CSV-timelocked pay-to-pubkey (OPoI escrow).
    /// Pattern: <seq_len> <seq_bytes[1..=8]> OP_CSV OpData32 <pubkey_32> OP_CHECKSIG
    #[inline(always)]
    pub fn is_csv_pay_to_pubkey(script_public_key: &[u8]) -> bool {
        let len = script_public_key.len();
        // seq_len in 1..=8, total = seq_len + 1(len_byte) + 1(CSV) + 1(OpData32) + 32(key) + 1(CHECKSIG)
        if len < 37 || len > 44 { return false; }
        let seq_len = script_public_key[0] as usize;
        if seq_len == 0 || seq_len > 8 { return false; }
        if len != seq_len + 36 { return false; }
        script_public_key[seq_len + 1] == opcodes::codes::OpCheckSequenceVerify
            && script_public_key[seq_len + 2] == opcodes::codes::OpData32
            && script_public_key[len - 1] == opcodes::codes::OpCheckSig
    }

    fn as_str(&self) -> &'static str {
        match self {
            ScriptClass::NonStandard => NON_STANDARD,
            ScriptClass::PubKey => PUB_KEY,
            ScriptClass::PubKeyECDSA => PUB_KEY_ECDSA,
            ScriptClass::ScriptHash => SCRIPT_HASH,
            ScriptClass::CsvPubKey => CSV_PUB_KEY,
        }
    }

    pub fn version(&self) -> ScriptPublicKeyVersion {
        match self {
            ScriptClass::NonStandard => 0,
            ScriptClass::PubKey => MAX_SCRIPT_PUBLIC_KEY_VERSION,
            ScriptClass::PubKeyECDSA => MAX_SCRIPT_PUBLIC_KEY_VERSION,
            ScriptClass::ScriptHash => MAX_SCRIPT_PUBLIC_KEY_VERSION,
            ScriptClass::CsvPubKey => MAX_SCRIPT_PUBLIC_KEY_VERSION,
        }
    }
}

impl Display for ScriptClass {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ScriptClass {
    type Err = Error;

    fn from_str(script_class: &str) -> Result<Self, Self::Err> {
        match script_class {
            NON_STANDARD => Ok(ScriptClass::NonStandard),
            PUB_KEY => Ok(ScriptClass::PubKey),
            PUB_KEY_ECDSA => Ok(ScriptClass::PubKeyECDSA),
            SCRIPT_HASH => Ok(ScriptClass::ScriptHash),
            CSV_PUB_KEY => Ok(ScriptClass::CsvPubKey),
            _ => Err(Error::InvalidScriptClass(script_class.to_string())),
        }
    }
}

impl TryFrom<&str> for ScriptClass {
    type Error = Error;

    fn try_from(script_class: &str) -> Result<Self, Self::Error> {
        script_class.parse()
    }
}

impl From<Version> for ScriptClass {
    fn from(value: Version) -> Self {
        match value {
            Version::PubKey => ScriptClass::PubKey,
            Version::PubKeyECDSA => ScriptClass::PubKeyECDSA,
            Version::ScriptHash => ScriptClass::ScriptHash,
        }
    }
}

#[cfg(test)]
mod tests {
    use keryx_consensus_core::tx::ScriptVec;
    use keryx_utils::hex::FromHex;

    use super::*;

    #[test]
    fn test_script_class_from_script() {
        struct Test {
            name: &'static str,
            script: Vec<u8>,
            version: ScriptPublicKeyVersion,
            class: ScriptClass,
        }

        // cspell:disable
        let tests = vec![
            Test {
                name: "valid pubkey script",
                script: Vec::from_hex("204a23f5eef4b2dead811c7efb4f1afbd8df845e804b6c36a4001fc096e13f8151ac").unwrap(),
                version: 0,
                class: ScriptClass::PubKey,
            },
            Test {
                name: "valid pubkey ecdsa script",
                script: Vec::from_hex("21fd4a23f5eef4b2dead811c7efb4f1afbd8df845e804b6c36a4001fc096e13f8151ab").unwrap(),
                version: 0,
                class: ScriptClass::PubKeyECDSA,
            },
            Test {
                name: "valid scripthash script",
                script: Vec::from_hex("aa204a23f5eef4b2dead811c7efb4f1afbd8df845e804b6c36a4001fc096e13f815187").unwrap(),
                version: 0,
                class: ScriptClass::ScriptHash,
            },
            Test {
                name: "non standard script (unexpected version)",
                script: Vec::from_hex("204a23f5eef4b2dead811c7efb4f1afbd8df845e804b6c36a4001fc096e13f8151ac").unwrap(),
                version: MAX_SCRIPT_PUBLIC_KEY_VERSION + 1,
                class: ScriptClass::NonStandard,
            },
            Test {
                name: "non standard script (unexpected key len)",
                script: Vec::from_hex("1f4a23f5eef4b2dead811c7efb4f1afbd8df845e804b6c36a4001fc096e13f81ac").unwrap(),
                version: 0,
                class: ScriptClass::NonStandard,
            },
            Test {
                name: "non standard script (unexpected final check sig op)",
                script: Vec::from_hex("204a23f5eef4b2dead811c7efb4f1afbd8df845e804b6c36a4001fc096e13f8151ad").unwrap(),
                version: 0,
                class: ScriptClass::NonStandard,
            },
        ];
        // cspell:enable

        for test in tests {
            let script_public_key = ScriptPublicKey::new(test.version, ScriptVec::from_iter(test.script.iter().copied()));
            assert_eq!(test.class, ScriptClass::from_script(&script_public_key), "{} wrong script class", test.name);
        }
    }
}
