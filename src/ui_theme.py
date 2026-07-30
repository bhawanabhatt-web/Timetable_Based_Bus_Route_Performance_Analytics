import tkinter as tk
from tkinter import ttk

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
NAVY_DARK = (10, 25, 66)      
STEEL_BLUE = (21, 96, 189)   
BG_APP = "#0f1b33"            # window background
BG_CARD = "#16294d"           # card background
BG_INPUT = "#1c2f57"          # entry/combobox background
FG_TEXT = "#e8f0ff"           # primary text
FG_MUTED = "#9fb3d9"          # secondary text
ACCENT = "#2f8fef"            # buttons / highlights
ACCENT_HOVER = "#4ba3ff"
BORDER = "#22375f"
DATA_PING = (56, 214, 214)    # cyan -- the trailing "data point" dots


def _hex(rgb):
    r, g, b = (max(0, min(255, int(v))) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def apply_theme(root: tk.Tk):
    """Configure ttk styles for a consistent dark navy/blue look."""
    root.configure(bg=BG_APP)
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure("TFrame", background=BG_APP)
    style.configure("Card.TFrame", background=BG_CARD, relief="flat")
    style.configure("TLabel", background=BG_APP, foreground=FG_TEXT, font=("Segoe UI", 10))
    style.configure("Card.TLabel", background=BG_CARD, foreground=FG_TEXT, font=("Segoe UI", 10))
    style.configure("Muted.TLabel", background=BG_APP, foreground=FG_MUTED, font=("Segoe UI", 9))
    style.configure("CardMuted.TLabel", background=BG_CARD, foreground=FG_MUTED, font=("Segoe UI", 9))
    style.configure("Heading.TLabel", background=BG_APP, foreground=FG_TEXT, font=("Segoe UI", 12, "bold"))
    style.configure("CardHeading.TLabel", background=BG_CARD, foreground=FG_TEXT, font=("Segoe UI", 11, "bold"))
    style.configure("Metric.TLabel", background=BG_CARD, foreground=ACCENT_HOVER, font=("Segoe UI", 20, "bold"))

    style.configure("TButton", background=ACCENT, foreground="white",
                    font=("Segoe UI", 10, "bold"), padding=8, borderwidth=0)
    style.map("TButton",
              background=[("active", ACCENT_HOVER), ("disabled", "#3a4a6b")],
              foreground=[("disabled", "#7d8cad")])

    style.configure("TNotebook", background=BG_APP, borderwidth=0, tabmargins=(6, 6, 6, 0))
    style.configure("TNotebook.Tab", background=BG_CARD, foreground=FG_MUTED,
                     padding=(16, 8), font=("Segoe UI", 10))
    style.map("TNotebook.Tab",
              background=[("selected", ACCENT)],
              foreground=[("selected", "white")])

    style.configure("TCombobox", fieldbackground=BG_INPUT, background=BG_INPUT,
                     foreground=FG_TEXT, arrowcolor=FG_TEXT)
    style.map("TCombobox", fieldbackground=[("readonly", BG_INPUT)])

    style.configure("TEntry", fieldbackground=BG_INPUT, foreground=FG_TEXT,
                     insertcolor=FG_TEXT, borderwidth=1)

    style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=BG_CARD,
                     borderwidth=0)

    style.configure("Vertical.TScrollbar", background=BG_CARD, troughcolor=BG_APP,
                     bordercolor=BG_APP, arrowcolor=FG_MUTED, relief="flat")
    style.map("Vertical.TScrollbar", background=[("active", ACCENT)])

    return style


class Card(ttk.Frame):
    """A padded, bordered container to group related widgets."""
    def __init__(self, master, title=None, **kwargs):
        super().__init__(master, style="Card.TFrame", padding=16, **kwargs)
        if title:
            ttk.Label(self, text=title, style="CardHeading.TLabel").pack(anchor="w", pady=(0, 10))


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable container. Add widgets to `.body`, not to
    the ScrollableFrame itself.

    Wheel scrolling is NOT bound automatically on hover (that breaks over
    embedded matplotlib canvases). Instead, call `.bind_wheel(root)` when
    this frame's tab becomes active, and it'll respond to the mouse wheel
    anywhere in the window until another ScrollableFrame calls
    `.bind_wheel()` in turn. See DashboardApp's <<NotebookTabChanged>>
    handler for the intended usage pattern.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, style="TFrame", **kwargs)
        self.canvas = tk.Canvas(self, bg=BG_APP, highlightthickness=0)
        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vscroll.pack(side="right", fill="y")

        self.body = ttk.Frame(self.canvas, style="TFrame")
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_body_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self._window, width=event.width)

    def bind_wheel(self, root):
        root.bind_all("<MouseWheel>", self._on_mousewheel)     # Windows/macOS
        root.bind_all("<Button-4>", self._on_mousewheel_linux)  # Linux scroll up
        root.bind_all("<Button-5>", self._on_mousewheel_linux)  # Linux scroll down

    def unbind_wheel(self, root):
        root.unbind_all("<MouseWheel>")
        root.unbind_all("<Button-4>")
        root.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")


class AnimatedBanner(tk.Canvas):
    """A gradient header banner with a small bus travelling along a dashed
    route line, leaving a fading trail of "data ping" dots behind it --
    themed on GPS-tracked bus data rather than a generic UI shine effect.
    """

    def __init__(self, master, title, subtitle="", height=92, **kwargs):
        super().__init__(master, height=height, highlightthickness=0,
                          bg=BG_APP, bd=0, **kwargs)
        self.title_text = title
        self.subtitle_text = subtitle
        self.banner_height = height
        self._running = True
        self._bus_x = -60.0
        self._trail = []       # list of [x, y, age]
        self._trail_life = 16
        self._tick = 0
        self.bind("<Configure>", lambda e: self._draw_static())
        self.after(50, self._draw_static)
        self.after(300, self._animate)

    def destroy(self):
        self._running = False
        super().destroy()

    def _bg_color_at(self, frac):
        t = max(0.0, min(1.0, frac))
        r = NAVY_DARK[0] + (STEEL_BLUE[0] - NAVY_DARK[0]) * t
        g = NAVY_DARK[1] + (STEEL_BLUE[1] - NAVY_DARK[1]) * t
        b = NAVY_DARK[2] + (STEEL_BLUE[2] - NAVY_DARK[2]) * t
        return (r, g, b)

    def _route_y(self):
        return self.banner_height - 20

    def _draw_static(self):
        self.delete("bg")
        w = max(self.winfo_width(), 400)
        h = self.banner_height
        steps = 60
        for i in range(steps):
            t = i / steps
            color = _hex(self._bg_color_at(t))
            x0, x1 = w * i / steps, w * (i + 1) / steps
            self.create_rectangle(x0, 0, x1, h, fill=color, outline=color, tags="bg")

        y_title = h // 2 - 18
        self.create_text(24, y_title, anchor="w", text=self.title_text,
                          font=("Segoe UI", 18, "bold"), fill="white", tags="bg")
        if self.subtitle_text:
            self.create_text(24, y_title + 26, anchor="w", text=self.subtitle_text,
                              font=("Segoe UI", 9), fill="#cfe1ff", tags="bg")

        # Dashed route line the bus travels along
        route_y = self._route_y()
        self.create_line(0, route_y, w, route_y, fill="#3d5f95", width=2,
                          dash=(7, 5), tags="bg")
        self.tag_raise("scene")

    def _blend_toward_bg(self, rgb, t, x_frac):
        base = self._bg_color_at(x_frac)
        r = rgb[0] + (base[0] - rgb[0]) * t
        g = rgb[1] + (base[1] - rgb[1]) * t
        b = rgb[2] + (base[2] - rgb[2]) * t
        return _hex((r, g, b))

    def _draw_bus(self, x, y):
        body_w, body_h = 36, 16
        # Shadow
        self.create_oval(x + 2, y + 4, x + body_w - 2, y + 9, fill="#08132b", outline="", tags="scene")
        # Body
        self.create_rectangle(x, y - body_h, x + body_w, y, fill="#f4f9ff",
                               outline="#0a1942", width=1.3, tags="scene")
        # Windows
        self.create_rectangle(x + 5, y - body_h + 3, x + body_w - 5, y - 7,
                               fill="#9fd6ff", outline="", tags="scene")
        # Wheels
        self.create_oval(x + 5, y - 3, x + 12, y + 4, fill="#0a1942", outline="", tags="scene")
        self.create_oval(x + body_w - 12, y - 3, x + body_w - 5, y + 4, fill="#0a1942", outline="", tags="scene")

    def _animate(self):
        if not self._running:
            return
        w = max(self.winfo_width(), 400)
        route_y = self._route_y()

        self.delete("scene")

        self._tick += 1
        if self._tick % 4 == 0:
            self._trail.append([self._bus_x, route_y, 0])
        self._trail = [[x, y, age + 1] for x, y, age in self._trail if age + 1 < self._trail_life]

        for x, y, age in self._trail:
            t = age / self._trail_life
            size = max(1.0, 4.5 * (1 - t))
            color = self._blend_toward_bg(DATA_PING, t, x / w if w else 0)
            self.create_oval(x - size, y - size, x + size, y + size,
                              fill=color, outline="", tags="scene")

        self._draw_bus(self._bus_x, route_y)

        self._bus_x += 2.4
        if self._bus_x > w + 60:
            self._bus_x = -60.0
            self._trail = []

        self.after(40, self._animate)
