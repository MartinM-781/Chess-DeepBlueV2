//! IA d'échecs par self-play : réseau de valeur (MLP maison) + moteur shakmaty.
//! Même philosophie que l'IA de poker (C:\dev\poker) : tout le chemin chaud compilé,
//! dashboard web (port 8778) avec plateau jouable et courbe d'entraînement,
//! adversaires à paliers de temps d'entraînement (1 h, 3 h, 10 h, ...).

/// Windows 11 classe les processus d'arrière-plan sans fenêtre en mode
/// « efficacité » (EcoQoS) et les confine sur les cœurs efficients — mesuré :
/// 7 cœurs servis sur 20 malgré 18 threads prêts. Cet appel déclare
/// explicitement « pas de throttling » (l'API des applications hautes
/// performances). Sans effet ailleurs que sous Windows.
#[cfg(windows)]
pub fn pleine_puissance() {
    #[repr(C)]
    struct PowerThrottlingState {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }
    extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn SetProcessInformation(
            h: *mut core::ffi::c_void,
            classe: i32,
            info: *mut core::ffi::c_void,
            taille: u32,
        ) -> i32;
    }
    // ProcessPowerThrottling = 4 ; EXECUTION_SPEED = 1 ; state_mask 0 = jamais.
    let mut etat = PowerThrottlingState { version: 1, control_mask: 1, state_mask: 0 };
    unsafe {
        SetProcessInformation(GetCurrentProcess(), 4, &mut etat as *mut _ as *mut _, 12);
    }
}

#[cfg(not(windows))]
pub fn pleine_puissance() {}

pub mod arena;
pub mod bots;
pub mod checkpoints;
pub mod elo;
pub mod features;
pub mod nn;
pub mod nnue;
pub mod search;
pub mod selfplay;
pub mod uci;
