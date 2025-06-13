use glm::Vec2;
use log::info;
use std::ffi::CString;
use windows::Win32::UI::WindowsAndMessaging::{RegisterClassA, WNDCLASSA};
use windows::{
    Win32::{
        Foundation::{HWND, LPARAM, LRESULT, WPARAM},
        System::LibraryLoader::GetModuleHandleA,
        UI::WindowsAndMessaging::*,
    },
    core::PCSTR,
};

pub struct Window {
    hwnd: HWND,
    window_title: String,
    is_running: bool,
    size: Vec2,
}

impl Window {
    pub fn new(window_title: &str) -> Self {
        let window_title = window_title.to_string();
        let is_running = true;

        let h_instance = unsafe { GetModuleHandleA(None) }.unwrap().into();
        let class_name = CString::new("window_class").unwrap();

        let window_class = WNDCLASSA {
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(window_callback),
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: h_instance,
            hCursor: unsafe { LoadCursorW(None, IDC_ARROW).unwrap() },
            lpszClassName: PCSTR(class_name.as_ptr() as _),
            ..Default::default()
        };

        let hwnd: HWND;

        unsafe {
            RegisterClassA(&window_class);

            hwnd = CreateWindowExA(
                Default::default(),
                PCSTR(class_name.as_ptr() as _),
                PCSTR(window_title.as_ptr() as _),
                WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                1280,
                720,
                None,
                None,
                Some(h_instance),
                None,
            )
            .expect("Failed to create window");
        }

        info!("Window created: {}", window_title);
        Self {
            hwnd,
            window_title,
            is_running,
            size: Vec2::new(1280.0, 720.0),
        }
    }

    pub fn register_window(&mut self) {
        unsafe {
            SetWindowLongPtrA(self.hwnd, GWLP_USERDATA, self as *mut _ as _);
        }
    }

    pub fn poll_events(&self) {
        unsafe {
            let mut msg = MSG::default();
            while PeekMessageA(&mut msg, Some(self.hwnd), 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageA(&msg);
            }
        }
    }

    pub fn set_title(&mut self, title: &str) {
        self.window_title = title.to_string();
        unsafe {
            SetWindowTextA(self.hwnd, PCSTR(title.as_ptr() as _))
                .expect("Failed to set window title");
        }
    }

    pub fn is_open(&self) -> bool {
        self.is_running
    }

    pub fn close_window(&mut self) {
        info!("Window closed: {}", self.window_title);
        self.is_running = false;
    }

    pub fn get_handle(&self) -> HWND {
        self.hwnd
    }

    pub fn get_title(&self) -> &str {
        &self.window_title
    }

    pub fn get_size(&self) -> Vec2 {
        self.size
    }
}

unsafe extern "system" fn window_callback(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    unsafe {
        let window_ptr = GetWindowLongPtrA(hwnd, GWLP_USERDATA) as *mut Window;
        match msg {
            WM_CLOSE => {
                PostQuitMessage(0);
                DestroyWindow(hwnd).expect("Failed to destroy window");
                LRESULT(0)
            }
            WM_DESTROY => {
                window_ptr.as_mut().unwrap().close_window();
                PostQuitMessage(0);
                LRESULT(0)
            }
            _ => DefWindowProcA(hwnd, msg, wparam, lparam),
        }
    }
}
