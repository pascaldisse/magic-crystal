//! Window input for the Embodiment (Rite E0): keyboard + mouse-look that drive
//! the [`Player`] in the native Scrying Glass window.
//!
//! macOS-only, matching this package's existing native click monitor: a single
//! `NSEvent` local monitor observes key up/down, modifier flags (Shift), and
//! relative mouse motion, translating them into [`Key`]
//! intents and look deltas on the shared player. Click captures the pointer
//! (cursor hidden + disassociated so mouse-look never hits a screen edge); Esc
//! releases it. Speeds and sensitivity live in
//! [`PlayerParams`](crate::player::PlayerParams) — nothing here is hardcoded
//! beyond the fixed macOS virtual key codes.

#[cfg(target_os = "macos")]
use crate::player::{Key, Player};
#[cfg(target_os = "macos")]
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

// macOS ANSI virtual key codes (Carbon `kVK_*`): stable identifiers, not tunables.
#[cfg(target_os = "macos")]
mod vk {
    pub const A: u16 = 0;
    pub const S: u16 = 1;
    pub const D: u16 = 2;
    pub const C: u16 = 8;
    pub const F: u16 = 3;
    pub const W: u16 = 13;
    pub const SPACE: u16 = 49;
    pub const ESCAPE: u16 = 53;
}

// Cursor association toggle from CoreGraphics: while disassociated, mouse
// motion still arrives as relative deltas but the cursor stops moving — real
// pointer capture for mouse-look.
#[cfg(target_os = "macos")]
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGAssociateMouseAndMouseCursorPosition(connected: bool) -> i32;
}

/// Map a macOS virtual key code to a movement intent, if any.
#[cfg(target_os = "macos")]
fn intent(code: u16) -> Option<Key> {
    match code {
        vk::W => Some(Key::Forward),
        vk::S => Some(Key::Back),
        vk::A => Some(Key::Left),
        vk::D => Some(Key::Right),
        vk::SPACE => Some(Key::Jump),
        vk::C => Some(Key::Crouch),
        _ => None,
    }
}

/// Install the window keyboard + mouse-look monitor. Drives `player` in place.
#[cfg(target_os = "macos")]
pub fn install_player_input(player: Arc<Mutex<Player>>) -> Result<(), String> {
    use block2::RcBlock;
    use objc2_app_kit::{NSCursor, NSEvent, NSEventMask, NSEventModifierFlags, NSEventType};

    let captured = Arc::new(AtomicBool::new(false));
    let mask = NSEventMask::KeyDown
        | NSEventMask::KeyUp
        | NSEventMask::FlagsChanged
        | NSEventMask::MouseMoved
        | NSEventMask::LeftMouseDragged
        | NSEventMask::LeftMouseDown;

    let block = RcBlock::new(move |event: std::ptr::NonNull<NSEvent>| -> *mut NSEvent {
        let event_ref = unsafe { event.as_ref() };
        let kind = event_ref.r#type();
        match kind {
            NSEventType::LeftMouseDown => {
                if !captured.swap(true, Ordering::AcqRel) {
                    unsafe {
                        CGAssociateMouseAndMouseCursorPosition(false);
                        NSCursor::hide();
                    }
                } else if let Ok(mut player) = player.lock() {
                    // PLAYGROUND — a click while ALREADY pointer-locked is a
                    // PUSH (the first click only captures the pointer).
                    player.push_pending = true;
                }
            }
            NSEventType::KeyDown => {
                let code = event_ref.keyCode();
                if code == vk::ESCAPE {
                    if captured.swap(false, Ordering::AcqRel) {
                        unsafe {
                            CGAssociateMouseAndMouseCursorPosition(true);
                            NSCursor::unhide();
                        }
                    }
                } else if code == vk::F {
                    // PLAYGROUND — F is the PUSH key: edge-fire a shove of the
                    // body the view ray is aimed at (consumed by the render
                    // loop, same Op::Impulse route an agent op would take).
                    if let Ok(mut player) = player.lock() {
                        player.push_pending = true;
                    }
                } else if let Some(key) = intent(code)
                    && let Ok(mut player) = player.lock()
                {
                    player.keys.insert(key);
                }
            }
            NSEventType::KeyUp => {
                if let Some(key) = intent(event_ref.keyCode())
                    && let Ok(mut player) = player.lock()
                {
                    player.keys.remove(&key);
                }
            }
            NSEventType::FlagsChanged => {
                // Shift is a modifier, not a key press — track it from the flags.
                let shift = event_ref
                    .modifierFlags()
                    .contains(NSEventModifierFlags::Shift);
                if let Ok(mut player) = player.lock() {
                    if shift {
                        player.keys.insert(Key::Run);
                    } else {
                        player.keys.remove(&Key::Run);
                    }
                }
            }
            NSEventType::MouseMoved | NSEventType::LeftMouseDragged => {
                if captured.load(Ordering::Acquire)
                    && let Ok(mut player) = player.lock()
                {
                    let dx = event_ref.deltaX() as f32;
                    let dy = event_ref.deltaY() as f32;
                    player.look(dx, dy);
                }
            }
            _ => {}
        }
        event_ref as *const NSEvent as *mut NSEvent
    });

    let monitor = unsafe { NSEvent::addLocalMonitorForEventsMatchingMask_handler(mask, &block) }
        .ok_or_else(|| "failed to install macOS player input monitor".to_string())?;
    Box::leak(Box::new(monitor));
    Ok(())
}

/// Non-macOS builds have no native window input yet — the `/walk` debug organ
/// is the portable path.
#[cfg(not(target_os = "macos"))]
pub fn install_player_input(
    _player: std::sync::Arc<std::sync::Mutex<crate::player::Player>>,
) -> Result<(), String> {
    Err("native window player input is macOS-only; use the /walk debug organ".into())
}
