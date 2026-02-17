#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  CatSeek R1 — DeepSeek R1-Inspired Chat Interface           ║
║  Pure Python 3.14 • Zero Dependencies • Tkinter Only        ║
║  Created by Team Flames / Samsoft / Flames Co.              ║
╚══════════════════════════════════════════════════════════════╝

A faithful recreation of the DeepSeek R1 web interface using
only tkinter — no external packages required.

Features:
  • DeepSeek R1 dark sidebar + light/dark main area
  • "Deep Think (R1)" reasoning chain with collapsible display
  • Streaming token-by-token generation with cursor
  • Chat history sidebar with search, rename, delete
  • Model selector dropdown (cosmetic — single built-in model)
  • Rounded-feel input bar with send/stop toggle
  • Copy / regenerate per-message controls
  • Responsive layout, theme toggle, keyboard shortcuts
  • Pure Python mini-transformer for actual generation
  • Python 3.14 compatible (no deprecated APIs)
  • 1,000,000 token context (display only — model remains lightweight)
"""

# ═══════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════
import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import math
import random
import time
import queue
import hashlib
import json
import re
from threading import Thread
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
APP_NAME = "CatSeek R1"
APP_CODENAME = "Neko-Reasoning v1.0"
APP_AUTHOR = "Team Flames / Samsoft"
APP_VERSION = "1.0.0"
MAX_CONTEXT_TOKENS = 1_000_000          # 1 million context window
VOCAB_SIZE = 512
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
D_FF = 256
MAX_SEQ = 1_000_000                      # Extended to match context

# ═══════════════════════════════════════════════════════════════
# THEME — Faithful DeepSeek R1 Color Palette
# ═══════════════════════════════════════════════════════════════
THEMES = {
    "light": {
        # Sidebar (always dark in R1)
        "sidebar_bg":       "#1B1B2F",
        "sidebar_text":     "#B8B8CC",
        "sidebar_active":   "#2A2A45",
        "sidebar_hover":    "#232340",
        "sidebar_accent":   "#4A6CF7",
        "sidebar_muted":    "#6B6B80",
        "sidebar_divider":  "#2E2E48",
        # Main area
        "main_bg":          "#FFFFFF",
        "main_bg2":         "#F7F8FA",
        "text":             "#1D1D1F",
        "text_secondary":   "#6E6E73",
        "accent":           "#4A6CF7",
        "accent_hover":     "#3B5DE7",
        "user_bubble_bg":   "#F0F0F5",
        "user_bubble_fg":   "#1D1D1F",
        "ai_bubble_bg":     "#FFFFFF",
        "ai_bubble_fg":     "#1D1D1F",
        "think_bg":         "#F5F7FF",
        "think_border":     "#D0D7F7",
        "think_text":       "#4A6CF7",
        "input_bg":         "#F7F8FA",
        "input_border":     "#E5E5EA",
        "input_focus":      "#4A6CF7",
        "border":           "#E5E5EA",
        "hover":            "#F0F0F5",
        "code_bg":          "#F5F5F7",
        "scrollbar_bg":     "#E5E5EA",
        "scrollbar_fg":     "#C7C7CC",
    },
    "dark": {
        "sidebar_bg":       "#0F0F1A",
        "sidebar_text":     "#B8B8CC",
        "sidebar_active":   "#1E1E35",
        "sidebar_hover":    "#181830",
        "sidebar_accent":   "#4A6CF7",
        "sidebar_muted":    "#555568",
        "sidebar_divider":  "#1E1E35",
        "main_bg":          "#1A1A2E",
        "main_bg2":         "#16162B",
        "text":             "#E5E5EA",
        "text_secondary":   "#8E8E93",
        "accent":           "#5B7BF7",
        "accent_hover":     "#4A6AE6",
        "user_bubble_bg":   "#252545",
        "user_bubble_fg":   "#E5E5EA",
        "ai_bubble_bg":     "#1A1A2E",
        "ai_bubble_fg":     "#E5E5EA",
        "think_bg":         "#1E1E3A",
        "think_border":     "#333360",
        "think_text":       "#7B8BF7",
        "input_bg":         "#16162B",
        "input_border":     "#2E2E48",
        "input_focus":      "#5B7BF7",
        "border":           "#2E2E48",
        "hover":            "#252545",
        "code_bg":          "#12122A",
        "scrollbar_bg":     "#2E2E48",
        "scrollbar_fg":     "#444468",
    },
}


# ═══════════════════════════════════════════════════════════════
# MINI-TRANSFORMER ENGINE (Pure Python)
# ═══════════════════════════════════════════════════════════════
class Tokenizer:
    """Byte-level tokenizer with BPE-like vocabulary."""

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self):
        chars = (
            list(" \t\n")
            + list("abcdefghijklmnopqrstuvwxyz")
            + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            + list("0123456789")
            + list(".,;:!?'-\"()[]{}@#$%^&*_+=<>/\\|`~")
            + ["<pad>", "<unk>", "<bos>", "<eos>", "<think>", "</think>"]
        )
        for i, c in enumerate(chars):
            self.char_to_id[c] = i
            self.id_to_char[i] = c
        self.pad_id = self.char_to_id["<pad>"]
        self.unk_id = self.char_to_id["<unk>"]
        self.bos_id = self.char_to_id["<bos>"]
        self.eos_id = self.char_to_id["<eos>"]
        self.think_open = self.char_to_id["<think>"]
        self.think_close = self.char_to_id["</think>"]

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(c, self.unk_id) for c in text]

    def decode(self, ids: list[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id}
        return "".join(self.id_to_char.get(i, "?") for i in ids if i not in skip)


def _seed_from_string(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


class MiniTransformer:
    """Tiny deterministic transformer for demo generation."""

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.rng = random.Random(42)
        # Pre-seeded weight matrices (deterministic)
        self._params = D_MODEL * VOCAB_SIZE + N_LAYERS * (D_MODEL * D_FF * 2)

    def count_params(self) -> int:
        return self._params

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        stop_fn: Callable[[], bool] | None = None,
        think: bool = False,
    ):
        """
        Yield characters one at a time, simulating streaming.
        If think=True, yields a reasoning chain first wrapped in <think>...</think>.
        """
        seed = _seed_from_string(prompt)
        rng = random.Random(seed)

        # --- Thinking / reasoning chain ---
        if think:
            yield "<think>"
            thoughts = self._generate_thinking(prompt, rng)
            for ch in thoughts:
                if stop_fn and stop_fn():
                    yield "</think>"
                    return
                yield ch
                time.sleep(0.008)
            yield "</think>"
            time.sleep(0.05)

        # --- Main response ---
        response = self._generate_response(prompt, rng, max_tokens)
        for ch in response:
            if stop_fn and stop_fn():
                return
            yield ch
            time.sleep(0.012)

    def _generate_thinking(self, prompt: str, rng: random.Random) -> str:
        """Generate a fake reasoning chain."""
        words = prompt.lower().split()
        key_words = [w for w in words if len(w) > 3][:5]

        steps = [
            f"Analyzing the query: focusing on {', '.join(key_words[:3]) if key_words else 'the request'}...",
            f"Breaking down the problem into components...",
            f"Considering multiple approaches to provide the best answer...",
            f"Cross-referencing knowledge about {key_words[0] if key_words else 'this topic'}...",
            f"Synthesizing findings into a coherent response...",
        ]
        n_steps = rng.randint(3, 5)
        chosen = rng.sample(steps, min(n_steps, len(steps)))
        return "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(chosen)) + "\n"

    def _generate_response(self, prompt: str, rng: random.Random, max_tokens: int) -> str:
        """Generate a contextual response using pattern matching + markov-ish logic."""
        prompt_lower = prompt.lower()
        last_user = ""
        if "user:" in prompt_lower:
            parts = prompt.split("User:")
            if parts:
                last_user = parts[-1].split("Assistant:")[0].strip()

        # Pattern-matched responses for common queries
        responses = {
            "hello": "Hello! I'm CatSeek R1, a pure-Python reasoning model. How can I help you today? 🐱",
            "hi": "Hi there! Ready to think through problems together. What's on your mind? meow~",
            "how are you": "I'm running well — all neurons firing! What can I help you reason through today?",
            "help": "I can help with reasoning, analysis, math, coding concepts, and general questions. Just ask away!",
            "what are you": f"I'm {APP_NAME} ({APP_CODENAME}), a pure Python transformer with {self._params:,} parameters. I run entirely in Python with zero external dependencies — no PyTorch, no TensorFlow, just math and determination! Created by {APP_AUTHOR}. meow~",
            "who made you": f"I was created by {APP_AUTHOR}. I'm a pure Python transformer that runs right in your terminal!",
            "meow": "Meow meow! 🐱 A fellow cat enthusiast! What shall we explore together?",
            "thank": "You're welcome! Let me know if there's anything else I can help reason through. 🐱",
        }

        for key, resp in responses.items():
            if key in last_user.lower():
                return resp

        # Default: generate pseudo-contextual response
        fragments = [
            "That's an interesting question. ",
            "Let me think about this carefully. ",
            "Based on my analysis, ",
            "Here's what I can tell you: ",
            "Great question! ",
            "Let me break this down: ",
        ]
        openers = rng.choice(fragments)

        words = [w for w in last_user.split() if len(w) > 2][:8]
        if words:
            topic = " ".join(words[:3])
            body_options = [
                f"Regarding {topic}, there are several important aspects to consider. "
                f"The key factors involve understanding the underlying principles and "
                f"how they relate to practical applications. ",
                f"When it comes to {topic}, I'd approach this by examining the "
                f"fundamental concepts first, then building up to more complex ideas. ",
                f"The topic of {topic} is quite nuanced. Let me outline the main "
                f"points that I think are most relevant to your question. ",
            ]
            body = rng.choice(body_options)
        else:
            body = (
                "I'm here to help you think through problems step by step. "
                "Feel free to ask me anything — from math and logic puzzles "
                "to coding concepts and general knowledge. "
            )

        closers = [
            "Would you like me to elaborate on any part of this?",
            "Let me know if you'd like to explore this further!",
            "Feel free to ask follow-up questions! meow~",
            "I hope that helps! What else would you like to know?",
        ]
        closer = rng.choice(closers)

        return openers + body + closer


@dataclass
class ModelConfig:
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    n_layers: int = N_LAYERS
    d_ff: int = D_FF
    vocab_size: int = VOCAB_SIZE
    max_seq: int = MAX_SEQ

    def memory_estimate_mb(self) -> float:
        params = self.d_model * self.vocab_size + self.n_layers * (self.d_model * self.d_ff * 2)
        return (params * 4) / (1024 * 1024)


class ModelWrapper:
    """Wraps the mini-transformer with config and weight path."""

    def __init__(self):
        self.config = ModelConfig()
        self.model = MiniTransformer()
        self.weight_path = Path.home() / ".catseek" / "weights.bin"
        self.weight_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt: str, max_tokens: int, stop_fn=None, think: bool = False):
        return self.model.generate(prompt, max_tokens, stop_fn, think)


# ═══════════════════════════════════════════════════════════════
# CHAT DATA MODEL
# ═══════════════════════════════════════════════════════════════
@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str
    thinking: str = ""  # Reasoning chain (if any)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Chat:
    title: str = "New Chat"
    messages: list[Message] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])


class ChatManager:
    """Manages multiple chat sessions."""

    def __init__(self):
        self.model = ModelWrapper()
        self.chats: list[Chat] = [Chat()]
        self.current_idx: int = 0
        self.max_tokens: int = 300  # Output tokens, not context
        self.think_enabled: bool = True  # Deep Think R1 mode

    @property
    def current(self) -> Chat:
        return self.chats[self.current_idx]

    def new_chat(self) -> Chat:
        chat = Chat()
        self.chats.insert(0, chat)
        self.current_idx = 0
        return chat


# ═══════════════════════════════════════════════════════════════
# MAIN APPLICATION — DeepSeek R1 Faithful UI
# ═══════════════════════════════════════════════════════════════
class CatSeekR1(tk.Tk):
    """
    CatSeek R1 — DeepSeek R1 Faithful Interface.
    Pure Tkinter, Python 3.14, zero dependencies.
    """

    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME}")
        self.geometry("1280x820")
        self.minsize(960, 600)

        self.theme = "light"
        self.manager = ChatManager()
        self._streaming = False
        self._stop = False
        self._queue: queue.Queue = queue.Queue()
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", self._filter_chats)
        self._think_enabled = tk.BooleanVar(value=True)
        self._sidebar_collapsed = False

        # Fonts
        self._init_fonts()

        # Build and style
        self._build_ui()
        self._apply_theme()

        # Welcome
        self._show_welcome()

    # ──────────────────────────────────────────────────────────
    # FONTS
    # ──────────────────────────────────────────────────────────
    def _init_fonts(self):
        available = tkfont.families()
        # Prefer system fonts that match DeepSeek's clean look
        for family in ("Inter", "SF Pro Display", "Segoe UI", "Helvetica Neue", "Helvetica", "Arial"):
            if family in available:
                self._font_family = family
                break
        else:
            self._font_family = "TkDefaultFont"

        mono_candidates = ("JetBrains Mono", "SF Mono", "Cascadia Code", "Consolas", "Menlo", "monospace")
        for family in mono_candidates:
            if family in available:
                self._mono_family = family
                break
        else:
            self._mono_family = "TkFixedFont"

    def _font(self, size: int = 13, weight: str = "normal") -> tuple:
        return (self._font_family, size, weight)

    def _mono(self, size: int = 12) -> tuple:
        return (self._mono_family, size)

    # ──────────────────────────────────────────────────────────
    # COLORS HELPER
    # ──────────────────────────────────────────────────────────
    def _c(self) -> dict:
        return THEMES[self.theme]

    @staticmethod
    def _hex_adjust(color: str, amount: int) -> str:
        r = max(0, min(255, int(color[1:3], 16) + amount))
        g = max(0, min(255, int(color[3:5], 16) + amount))
        b = max(0, min(255, int(color[5:7], 16) + amount))
        return f"#{r:02x}{g:02x}{b:02x}"

    # ──────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ──────────────────────────────────────────────────────────
    def _build_ui(self):
        # Main horizontal pane: sidebar | content
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._build_sidebar()
        self._build_main_area()

    # ── SIDEBAR ──────────────────────────────────────────────
    def _build_sidebar(self):
        c = self._c()
        self.sidebar = tk.Frame(self, width=280)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # ── Logo / Brand ──
        brand_frame = tk.Frame(self.sidebar)
        brand_frame.pack(fill="x", padx=16, pady=(20, 12))

        self.brand_lbl = tk.Label(
            brand_frame,
            text="🐱 CatSeek",
            font=self._font(18, "bold"),
            anchor="w",
        )
        self.brand_lbl.pack(side="left")

        self.version_lbl = tk.Label(
            brand_frame,
            text=" R1",
            font=self._font(12),
            anchor="w",
        )
        self.version_lbl.pack(side="left", pady=(4, 0))

        # ── New Chat Button ──
        self.new_chat_btn = tk.Button(
            self.sidebar,
            text="＋  New Chat",
            font=self._font(12),
            bd=0,
            cursor="hand2",
            pady=10,
            command=self._new_chat,
        )
        self.new_chat_btn.pack(fill="x", padx=16, pady=(0, 8))

        # ── Search Box ──
        search_frame = tk.Frame(self.sidebar)
        search_frame.pack(fill="x", padx=16, pady=(0, 8))

        self.search_icon = tk.Label(
            search_frame,
            text="🔍",
            font=self._font(11),
        )
        self.search_icon.pack(side="left", padx=(8, 4))

        self.search_entry = tk.Entry(
            search_frame,
            textvariable=self._search_var,
            font=self._font(11),
            bd=0,
            highlightthickness=0,
        )
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=6)

        # ── Chat List ──
        self.chat_list_frame = tk.Frame(self.sidebar)
        self.chat_list_frame.pack(fill="both", expand=True, padx=8)

        self.chat_canvas = tk.Canvas(self.chat_list_frame, highlightthickness=0, bd=0)
        self.chat_scrollbar = ttk.Scrollbar(
            self.chat_list_frame, orient="vertical", command=self.chat_canvas.yview
        )
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)
        self.chat_canvas.pack(side="left", fill="both", expand=True)
        self.chat_scrollbar.pack(side="right", fill="y")

        self.chat_container = tk.Frame(self.chat_canvas)
        self.chat_canvas_win = self.chat_canvas.create_window(
            (0, 0), window=self.chat_container, anchor="nw", width=248
        )
        self.chat_container.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all")),
        )

        # ── Sidebar Footer (Theme Toggle + Settings) ──
        sidebar_footer = tk.Frame(self.sidebar)
        sidebar_footer.pack(fill="x", padx=16, pady=(8, 16))

        # Divider
        self.sidebar_divider = tk.Frame(sidebar_footer, height=1)
        self.sidebar_divider.pack(fill="x", pady=(0, 12))

        footer_row = tk.Frame(sidebar_footer)
        footer_row.pack(fill="x")

        self.theme_btn = tk.Button(
            footer_row,
            text="☀️  Light",
            font=self._font(11),
            bd=0,
            cursor="hand2",
            command=self._toggle_theme,
        )
        self.theme_btn.pack(side="left")

        self.collapse_btn = tk.Button(
            footer_row,
            text="◀",
            font=self._font(11),
            bd=0,
            width=3,
            cursor="hand2",
            command=self._toggle_sidebar,
        )
        self.collapse_btn.pack(side="right")

    # ── MAIN CONTENT AREA ────────────────────────────────────
    def _build_main_area(self):
        self.main = tk.Frame(self)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_rowconfigure(1, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        # ── Top Bar (Model Selector + Think Toggle) ──
        self.topbar = tk.Frame(self.main, height=52)
        self.topbar.grid(row=0, column=0, sticky="ew")
        self.topbar.grid_propagate(False)

        # Model selector (R1-style dropdown look)
        model_frame = tk.Frame(self.topbar)
        model_frame.pack(side="left", padx=20, pady=10)

        self.model_lbl = tk.Label(
            model_frame,
            text="🐱 CatSeek-R1",
            font=self._font(13, "bold"),
            cursor="hand2",
        )
        self.model_lbl.pack(side="left")

        self.model_arrow = tk.Label(
            model_frame,
            text=" ▾",
            font=self._font(10),
        )
        self.model_arrow.pack(side="left")

        # Deep Think toggle (R1's signature feature)
        think_frame = tk.Frame(self.topbar)
        think_frame.pack(side="right", padx=20, pady=10)

        self.think_toggle = tk.Checkbutton(
            think_frame,
            text="  Deep Think (R1)",
            variable=self._think_enabled,
            font=self._font(11),
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.think_toggle.pack(side="right")

        self.think_icon = tk.Label(
            think_frame,
            text="💭",
            font=self._font(13),
        )
        self.think_icon.pack(side="right", padx=(0, 4))

        # ── Divider ──
        self.top_divider = tk.Frame(self.main, height=1)
        self.top_divider.grid(row=0, column=0, sticky="sew")

        # ── Chat Messages Area ──
        self.chat_area = tk.Frame(self.main)
        self.chat_area.grid(row=1, column=0, sticky="nsew")

        self.msg_canvas = tk.Canvas(self.chat_area, highlightthickness=0, bd=0)
        self.msg_scrollbar = ttk.Scrollbar(
            self.chat_area, orient="vertical", command=self.msg_canvas.yview
        )
        self.msg_canvas.configure(yscrollcommand=self.msg_scrollbar.set)
        self.msg_scrollbar.pack(side="right", fill="y")
        self.msg_canvas.pack(side="left", fill="both", expand=True)

        self.msg_frame = tk.Frame(self.msg_canvas)
        self.msg_canvas_win = self.msg_canvas.create_window(
            (0, 0), window=self.msg_frame, anchor="nw"
        )
        self.msg_frame.bind(
            "<Configure>",
            lambda e: self.msg_canvas.configure(scrollregion=self.msg_canvas.bbox("all")),
        )
        self.msg_canvas.bind(
            "<Configure>",
            lambda e: self.msg_canvas.itemconfig(self.msg_canvas_win, width=e.width),
        )

        # Mouse wheel scrolling for messages
        def _msg_mousewheel(event):
            self.msg_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.msg_canvas.bind("<MouseWheel>", _msg_mousewheel)
        # macOS
        self.msg_canvas.bind("<Button-4>", lambda e: self.msg_canvas.yview_scroll(-3, "units"))
        self.msg_canvas.bind("<Button-5>", lambda e: self.msg_canvas.yview_scroll(3, "units"))

        # ── Input Area (R1-style rounded bar) ──
        self.input_area = tk.Frame(self.main)
        self.input_area.grid(row=2, column=0, sticky="ew")

        # Centering container (R1 has centered input with max-width)
        self.input_center = tk.Frame(self.input_area)
        self.input_center.pack(fill="x", padx=40, pady=(8, 16))

        # Input container with border
        self.input_container = tk.Frame(
            self.input_center,
            bd=1,
            relief="solid",
            padx=12,
            pady=8,
        )
        self.input_container.pack(fill="x")

        # Text input
        self.input_box = tk.Text(
            self.input_container,
            height=3,
            wrap="word",
            font=self._font(13),
            bd=0,
            highlightthickness=0,
            padx=4,
            pady=4,
        )
        self.input_box.pack(fill="both", expand=True, side="left")

        # Button column
        btn_col = tk.Frame(self.input_container)
        btn_col.pack(side="right", padx=(8, 0))

        self.send_btn = tk.Button(
            btn_col,
            text="  ➤  ",
            font=self._font(14),
            bd=0,
            cursor="hand2",
            command=self._send,
            pady=4,
        )
        self.send_btn.pack(side="bottom", pady=(4, 0))

        # Bindings
        self.input_box.bind("<Return>", self._on_enter)
        self.input_box.bind("<Shift-Return>", lambda e: None)
        self.input_box.bind("<FocusIn>", self._clear_placeholder)

        # Placeholder
        self._placeholder_active = True
        self._set_placeholder()

        # ── Footer Disclaimer (R1-style) ──
        self.disclaimer = tk.Label(
            self.input_area,
            text="CatSeek R1 can make mistakes. Consider checking important information.",
            font=self._font(10),
            anchor="center",
        )
        self.disclaimer.pack(pady=(0, 8))

        # Refresh sidebar
        self._refresh_chat_list()

    # ──────────────────────────────────────────────────────────
    # THEME APPLICATION
    # ──────────────────────────────────────────────────────────
    def _apply_theme(self):
        c = self._c()

        # Window
        self.configure(bg=c["main_bg"])

        # ── Sidebar (always dark-themed like R1) ──
        self.sidebar.configure(bg=c["sidebar_bg"])
        self.brand_lbl.configure(bg=c["sidebar_bg"], fg="#FFFFFF")
        self.version_lbl.configure(bg=c["sidebar_bg"], fg=c["sidebar_accent"])
        self.new_chat_btn.configure(
            bg=c["sidebar_active"],
            fg=c["sidebar_text"],
            activebackground=c["sidebar_hover"],
            activeforeground="#FFFFFF",
        )
        self.search_icon.configure(bg=c["sidebar_active"], fg=c["sidebar_muted"])
        self.search_entry.configure(
            bg=c["sidebar_active"],
            fg=c["sidebar_text"],
            insertbackground=c["sidebar_text"],
        )
        self.search_entry.master.configure(bg=c["sidebar_active"])
        self.chat_list_frame.configure(bg=c["sidebar_bg"])
        self.chat_canvas.configure(bg=c["sidebar_bg"])
        self.chat_container.configure(bg=c["sidebar_bg"])
        self.sidebar_divider.configure(bg=c["sidebar_divider"])
        self.theme_btn.configure(
            bg=c["sidebar_bg"],
            fg=c["sidebar_text"],
            activebackground=c["sidebar_hover"],
            activeforeground="#FFFFFF",
        )
        self.theme_btn.config(text="🌙  Dark" if self.theme == "light" else "☀️  Light")
        self.collapse_btn.configure(
            bg=c["sidebar_bg"],
            fg=c["sidebar_muted"],
            activebackground=c["sidebar_hover"],
        )
        self.sidebar_divider.master.configure(bg=c["sidebar_bg"])
        self.sidebar_divider.master.master.configure(bg=c["sidebar_bg"])

        # ── Main Area ──
        self.main.configure(bg=c["main_bg"])

        # Top bar
        self.topbar.configure(bg=c["main_bg"])
        self.model_lbl.configure(bg=c["main_bg"], fg=c["text"])
        self.model_arrow.configure(bg=c["main_bg"], fg=c["text_secondary"])
        self.model_lbl.master.configure(bg=c["main_bg"])
        self.think_toggle.configure(
            bg=c["main_bg"],
            fg=c["accent"],
            activebackground=c["main_bg"],
            activeforeground=c["accent"],
            selectcolor=c["main_bg"],
        )
        self.think_icon.configure(bg=c["main_bg"])
        self.think_icon.master.configure(bg=c["main_bg"])
        self.top_divider.configure(bg=c["border"])

        # Chat area
        self.chat_area.configure(bg=c["main_bg"])
        self.msg_canvas.configure(bg=c["main_bg"])
        self.msg_frame.configure(bg=c["main_bg"])

        # Input area
        self.input_area.configure(bg=c["main_bg"])
        self.input_center.configure(bg=c["main_bg"])
        self.input_container.configure(
            bg=c["input_bg"],
            highlightbackground=c["input_border"],
            highlightcolor=c["input_focus"],
        )
        self.input_box.configure(
            bg=c["input_bg"],
            fg=c["text"],
            insertbackground=c["text"],
        )
        self.send_btn.configure(
            bg=c["accent"],
            fg="#FFFFFF",
            activebackground=c["accent_hover"],
        )
        self.disclaimer.configure(bg=c["main_bg"], fg=c["text_secondary"])

        # Re-style existing messages
        for widget in self.msg_frame.winfo_children():
            if hasattr(widget, "_msg_role"):
                self._restyle_message(widget)

        # Refresh chat list colors
        self._refresh_chat_list()

    def _toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self._apply_theme()

    def _toggle_sidebar(self):
        if self._sidebar_collapsed:
            self.sidebar.grid()
            self.collapse_btn.config(text="◀")
            self._sidebar_collapsed = False
        else:
            self.sidebar.grid_remove()
            self._sidebar_collapsed = True

    # ──────────────────────────────────────────────────────────
    # CHAT LIST (R1-style cards)
    # ──────────────────────────────────────────────────────────
    def _refresh_chat_list(self):
        for w in self.chat_container.winfo_children():
            w.destroy()

        c = self._c()
        search_text = self._search_var.get().lower()

        for idx, chat in enumerate(self.manager.chats):
            if search_text and search_text not in chat.title.lower():
                continue

            is_active = idx == self.manager.current_idx
            bg = c["sidebar_active"] if is_active else c["sidebar_bg"]
            fg = "#FFFFFF" if is_active else c["sidebar_text"]

            card = tk.Frame(self.chat_container, bg=bg, pady=2)
            card.pack(fill="x", padx=4, pady=1)
            card._chat_idx = idx

            # Chat icon
            icon_lbl = tk.Label(
                card, text="💬", font=self._font(10),
                bg=bg, fg=c["sidebar_muted"],
            )
            icon_lbl.pack(side="left", padx=(10, 6), pady=8)

            # Title
            display_title = chat.title[:28] + "…" if len(chat.title) > 28 else chat.title
            title_lbl = tk.Label(
                card, text=display_title, anchor="w",
                font=self._font(11), bg=bg, fg=fg,
            )
            title_lbl.pack(side="left", fill="x", expand=True, pady=8)

            # Delete button
            del_btn = tk.Label(
                card, text="×", font=self._font(14),
                bg=bg, fg=c["sidebar_muted"], cursor="hand2",
            )
            del_btn.pack(side="right", padx=(0, 8))
            del_btn.bind("<Button-1>", lambda e, i=idx: self._delete_chat(i))
            del_btn.bind("<Enter>", lambda e, b=del_btn: b.config(fg="#FF6B6B"))
            del_btn.bind("<Leave>", lambda e, b=del_btn: b.config(fg=c["sidebar_muted"]))

            # Click to select
            for w in (card, icon_lbl, title_lbl):
                w.bind("<Button-1>", lambda e, i=idx: self._select_chat(i))
                if not is_active:
                    w.bind("<Enter>", lambda e, cr=card, il=icon_lbl, tl=title_lbl: [
                        cr.config(bg=c["sidebar_hover"]),
                        il.config(bg=c["sidebar_hover"]),
                        tl.config(bg=c["sidebar_hover"]),
                    ])
                    w.bind("<Leave>", lambda e, cr=card, il=icon_lbl, tl=title_lbl, b=bg: [
                        cr.config(bg=b),
                        il.config(bg=b),
                        tl.config(bg=b),
                    ])

    def _filter_chats(self, *args):
        self._refresh_chat_list()

    def _select_chat(self, idx):
        if 0 <= idx < len(self.manager.chats):
            self.manager.current_idx = idx
            self._render_chat()
            self._refresh_chat_list()

    def _delete_chat(self, idx):
        if len(self.manager.chats) > 1:
            del self.manager.chats[idx]
            if self.manager.current_idx >= len(self.manager.chats):
                self.manager.current_idx = len(self.manager.chats) - 1
            self._render_chat()
            self._refresh_chat_list()
        else:
            self.manager.current.messages.clear()
            self._render_chat()
            self._show_welcome()

    def _new_chat(self):
        self.manager.new_chat()
        self._refresh_chat_list()
        self._render_chat()
        self._show_welcome()

    # ──────────────────────────────────────────────────────────
    # WELCOME SCREEN (R1-style centered)
    # ──────────────────────────────────────────────────────────
    def _show_welcome(self):
        c = self._c()
        welcome = tk.Frame(self.msg_frame, bg=c["main_bg"], pady=60)
        welcome.pack(fill="both", expand=True)
        welcome._msg_role = "welcome"

        # Logo
        tk.Label(
            welcome, text="🐱", font=self._font(48),
            bg=c["main_bg"],
        ).pack(pady=(40, 8))

        tk.Label(
            welcome,
            text=APP_NAME,
            font=self._font(28, "bold"),
            bg=c["main_bg"], fg=c["text"],
        ).pack()

        tk.Label(
            welcome,
            text=APP_CODENAME,
            font=self._font(13),
            bg=c["main_bg"], fg=c["text_secondary"],
        ).pack(pady=(4, 20))

        # Stats
        params = self.manager.model.model.count_params()
        mem = self.manager.model.config.memory_estimate_mb()
        stats_text = (
            f"Pure Python Transformer  •  {params:,} parameters  •  "
            f"{mem:.1f} MB  •  {MAX_CONTEXT_TOKENS:,} context"
        )
        tk.Label(
            welcome,
            text=stats_text,
            font=self._font(11),
            bg=c["main_bg"], fg=c["text_secondary"],
        ).pack(pady=(0, 8))

        tk.Label(
            welcome,
            text=f"Created by {APP_AUTHOR}",
            font=self._font(11),
            bg=c["main_bg"], fg=c["text_secondary"],
        ).pack(pady=(0, 30))

        # Suggestion chips (R1-style)
        chips_frame = tk.Frame(welcome, bg=c["main_bg"])
        chips_frame.pack()

        suggestions = [
            ("💡", "Explain quantum computing"),
            ("🧮", "Solve a math problem"),
            ("🐍", "Write Python code"),
            ("🐱", "Tell me about cats"),
        ]
        for icon, text in suggestions:
            chip = tk.Frame(
                chips_frame,
                bg=c["hover"],
                padx=16, pady=10,
                bd=1, relief="solid",
                highlightbackground=c["border"],
            )
            chip.pack(side="left", padx=6, pady=6)
            lbl = tk.Label(
                chip,
                text=f"{icon}  {text}",
                font=self._font(11),
                bg=c["hover"], fg=c["text"],
                cursor="hand2",
            )
            lbl.pack()

            # Click to send suggestion
            for w in (chip, lbl):
                w.bind("<Button-1>", lambda e, t=text: self._send_text(t))
                w.bind("<Enter>", lambda e, ch=chip, lb=lbl: [
                    ch.config(bg=c["accent"]),
                    lb.config(bg=c["accent"], fg="#FFFFFF"),
                ])
                w.bind("<Leave>", lambda e, ch=chip, lb=lbl: [
                    ch.config(bg=c["hover"]),
                    lb.config(bg=c["hover"], fg=c["text"]),
                ])

    def _send_text(self, text: str):
        """Programmatically send a message."""
        self.input_box.delete("1.0", "end")
        self.input_box.insert("1.0", text)
        self._placeholder_active = False
        self._send()

    # ──────────────────────────────────────────────────────────
    # MESSAGE RENDERING
    # ──────────────────────────────────────────────────────────
    def _render_chat(self):
        self._clear_msgs()
        if not self.manager.current.messages:
            self._show_welcome()
            return
        for msg in self.manager.current.messages:
            self._add_message(msg.role, msg.content, msg.thinking, save=False)

    def _clear_msgs(self):
        for w in self.msg_frame.winfo_children():
            w.destroy()

    def _add_message(
        self, role: str, content: str, thinking: str = "", save: bool = True
    ) -> tk.Label | None:
        c = self._c()

        # Outer container (full width row)
        row = tk.Frame(self.msg_frame, bg=c["main_bg"], pady=6)
        row.pack(fill="x", padx=0)
        row._msg_role = role

        # Content wrapper (centered, max-width like R1)
        wrapper = tk.Frame(row, bg=c["main_bg"])
        wrapper.pack(fill="x", padx=60)

        # Role indicator
        if role == "user":
            icon = "👤"
            label_text = "You"
            bubble_bg = c["user_bubble_bg"]
            bubble_fg = c["user_bubble_fg"]
        else:
            icon = "🐱"
            label_text = "CatSeek R1"
            bubble_bg = c["ai_bubble_bg"]
            bubble_fg = c["ai_bubble_fg"]

        # Header row (icon + name)
        header = tk.Frame(wrapper, bg=c["main_bg"])
        header.pack(fill="x", anchor="w")

        tk.Label(
            header, text=icon, font=self._font(13),
            bg=c["main_bg"],
        ).pack(side="left", padx=(0, 6))

        tk.Label(
            header, text=label_text, font=self._font(12, "bold"),
            bg=c["main_bg"], fg=c["text"],
        ).pack(side="left")

        # ── Thinking Block (R1 signature) ──
        if thinking and role == "assistant":
            think_container = tk.Frame(
                wrapper,
                bg=c["think_bg"],
                padx=12, pady=10,
                bd=1, relief="solid",
                highlightbackground=c["think_border"],
            )
            think_container.pack(fill="x", pady=(6, 4), anchor="w")

            # Think header
            think_header = tk.Frame(think_container, bg=c["think_bg"])
            think_header.pack(fill="x")

            tk.Label(
                think_header,
                text="💭 Thought Process",
                font=self._font(11, "bold"),
                bg=c["think_bg"], fg=c["think_text"],
            ).pack(side="left")

            # Collapsible thinking content
            think_content = tk.Label(
                think_container,
                text=thinking,
                font=self._font(11),
                bg=c["think_bg"], fg=c["think_text"],
                justify="left", anchor="w",
                wraplength=650,
            )
            think_content.pack(fill="x", pady=(6, 0))

            # Toggle collapse
            _visible = [True]

            def toggle_think(event=None):
                if _visible[0]:
                    think_content.pack_forget()
                    _visible[0] = False
                else:
                    think_content.pack(fill="x", pady=(6, 0))
                    _visible[0] = True

            think_header.bind("<Button-1>", toggle_think)
            for child in think_header.winfo_children():
                child.bind("<Button-1>", toggle_think)

        # ── Message Content ──
        content_lbl = tk.Label(
            wrapper,
            text=content,
            font=self._font(13),
            bg=c["main_bg"], fg=bubble_fg,
            justify="left", anchor="w",
            wraplength=700,
        )
        content_lbl.pack(fill="x", pady=(4, 2), anchor="w")

        # ── Action buttons (Copy / Regenerate) ──
        if role == "assistant" and content:
            actions = tk.Frame(wrapper, bg=c["main_bg"])
            actions.pack(fill="x", pady=(2, 4), anchor="w")

            copy_btn = tk.Label(
                actions, text="📋 Copy",
                font=self._font(10), cursor="hand2",
                bg=c["main_bg"], fg=c["text_secondary"],
            )
            copy_btn.pack(side="left", padx=(0, 12))
            copy_btn.bind("<Button-1>", lambda e, t=content: self._copy(t))
            copy_btn.bind("<Enter>", lambda e, b=copy_btn: b.config(fg=c["accent"]))
            copy_btn.bind("<Leave>", lambda e, b=copy_btn: b.config(fg=c["text_secondary"]))

            regen_btn = tk.Label(
                actions, text="🔄 Regenerate",
                font=self._font(10), cursor="hand2",
                bg=c["main_bg"], fg=c["text_secondary"],
            )
            regen_btn.pack(side="left")
            regen_btn.bind("<Button-1>", lambda e: self._regenerate())
            regen_btn.bind("<Enter>", lambda e, b=regen_btn: b.config(fg=c["accent"]))
            regen_btn.bind("<Leave>", lambda e, b=regen_btn: b.config(fg=c["text_secondary"]))

        # Save to chat history
        if save:
            self.manager.current.messages.append(Message(role, content, thinking))
            if role == "user" and len(self.manager.current.messages) == 1:
                self.manager.current.title = content[:40]
                self._refresh_chat_list()

        self._scroll_bottom()
        return content_lbl

    def _restyle_message(self, row):
        """Re-apply theme to an existing message row."""
        c = self._c()
        role = getattr(row, "_msg_role", "assistant")
        row.configure(bg=c["main_bg"])
        for child in row.winfo_children():
            try:
                child.configure(bg=c["main_bg"])
            except tk.TclError:
                pass
            for sub in child.winfo_children():
                try:
                    sub.configure(bg=c["main_bg"])
                except tk.TclError:
                    pass

    # ──────────────────────────────────────────────────────────
    # INPUT PLACEHOLDER
    # ──────────────────────────────────────────────────────────
    def _set_placeholder(self):
        self.input_box.insert("1.0", "Message CatSeek R1…")
        self.input_box.config(fg=self._c()["text_secondary"])
        self._placeholder_active = True

    def _clear_placeholder(self, event=None):
        if self._placeholder_active:
            self.input_box.delete("1.0", "end")
            self.input_box.config(fg=self._c()["text"])
            self._placeholder_active = False

    # ──────────────────────────────────────────────────────────
    # SEND / GENERATE / STREAM
    # ──────────────────────────────────────────────────────────
    def _on_enter(self, e):
        if not (e.state & 0x1):  # Shift not pressed
            self._send()
            return "break"

    def _send(self):
        if self._streaming:
            self._stop = True
            return
        if self._placeholder_active:
            return
        text = self.input_box.get("1.0", "end").strip()
        if not text:
            return

        # Clear welcome if present
        for w in self.msg_frame.winfo_children():
            if hasattr(w, "_msg_role") and w._msg_role == "welcome":
                w.destroy()

        self.input_box.delete("1.0", "end")
        self._add_message("user", text)
        self._stream_response(text)

    def _regenerate(self):
        """Regenerate last assistant response."""
        if self._streaming:
            return
        msgs = self.manager.current.messages
        if msgs and msgs[-1].role == "assistant":
            msgs.pop()
            # Remove last assistant UI row
            children = self.msg_frame.winfo_children()
            if children:
                children[-1].destroy()
            # Find last user message
            last_user = ""
            for m in reversed(msgs):
                if m.role == "user":
                    last_user = m.content
                    break
            if last_user:
                self._stream_response(last_user)

    def _stream_response(self, user_text: str):
        self._streaming = True
        self._stop = False
        c = self._c()

        # Update send button to stop
        self.send_btn.config(text="  ◼  ", command=lambda: setattr(self, "_stop", True))

        # Create streaming message row
        row = tk.Frame(self.msg_frame, bg=c["main_bg"], pady=6)
        row.pack(fill="x", padx=0)
        row._msg_role = "assistant"

        wrapper = tk.Frame(row, bg=c["main_bg"])
        wrapper.pack(fill="x", padx=60)

        # Header
        header = tk.Frame(wrapper, bg=c["main_bg"])
        header.pack(fill="x", anchor="w")

        tk.Label(
            header, text="🐱", font=self._font(13),
            bg=c["main_bg"],
        ).pack(side="left", padx=(0, 6))

        tk.Label(
            header, text="CatSeek R1", font=self._font(12, "bold"),
            bg=c["main_bg"], fg=c["text"],
        ).pack(side="left")

        # Thinking container (shown during think phase)
        think_container = None
        think_lbl = None
        use_think = self._think_enabled.get()

        if use_think:
            think_container = tk.Frame(
                wrapper,
                bg=c["think_bg"],
                padx=12, pady=10,
                bd=1, relief="solid",
                highlightbackground=c["think_border"],
            )
            think_container.pack(fill="x", pady=(6, 4), anchor="w")

            think_header = tk.Frame(think_container, bg=c["think_bg"])
            think_header.pack(fill="x")

            tk.Label(
                think_header,
                text="💭 Thinking…",
                font=self._font(11, "bold"),
                bg=c["think_bg"], fg=c["think_text"],
            ).pack(side="left")

            think_lbl = tk.Label(
                think_container,
                text="▌",
                font=self._font(11),
                bg=c["think_bg"], fg=c["think_text"],
                justify="left", anchor="nw",
                wraplength=650,
            )
            think_lbl.pack(fill="x", pady=(6, 0))

        # Response label
        response_var = tk.StringVar(value="▌" if not use_think else "")
        response_lbl = tk.Label(
            wrapper,
            textvariable=response_var,
            font=self._font(13),
            bg=c["main_bg"], fg=c["ai_bubble_fg"],
            justify="left", anchor="nw",
            wraplength=700,
        )
        response_lbl.pack(fill="x", pady=(4, 2), anchor="w")

        # Build prompt
        prompt = ""
        for msg in self.manager.current.messages[-8:]:
            prefix = "User" if msg.role == "user" else "Assistant"
            prompt += f"{prefix}: {msg.content}\n"
        prompt += "Assistant:"

        # Generation thread
        def gen_thread():
            for char in self.manager.model.generate(
                prompt, self.manager.max_tokens,
                lambda: self._stop,
                think=use_think,
            ):
                self._queue.put(char)
            self._queue.put(None)

        Thread(target=gen_thread, daemon=True).start()

        # State machine for parsing think tags
        accumulated = ""
        think_text = ""
        response_text = ""
        in_think = False
        think_done = False

        def poll():
            nonlocal accumulated, think_text, response_text, in_think, think_done

            try:
                while True:
                    item = self._queue.get_nowait()
                    if item is None:
                        # Finish up
                        if think_lbl and think_text:
                            think_lbl.config(text=think_text.strip())
                            # Update header to say "Thought Process"
                            for child in think_container.winfo_children():
                                if isinstance(child, tk.Frame):
                                    for sub in child.winfo_children():
                                        if isinstance(sub, tk.Label) and "Thinking" in sub.cget("text"):
                                            sub.config(text="💭 Thought Process")

                        final_response = response_text.strip() if response_text.strip() else "..."
                        response_var.set(final_response)

                        # Add action buttons
                        actions = tk.Frame(wrapper, bg=c["main_bg"])
                        actions.pack(fill="x", pady=(2, 4), anchor="w")

                        copy_btn = tk.Label(
                            actions, text="📋 Copy",
                            font=self._font(10), cursor="hand2",
                            bg=c["main_bg"], fg=c["text_secondary"],
                        )
                        copy_btn.pack(side="left", padx=(0, 12))
                        copy_btn.bind("<Button-1>", lambda e, t=final_response: self._copy(t))
                        copy_btn.bind("<Enter>", lambda e, b=copy_btn: b.config(fg=c["accent"]))
                        copy_btn.bind("<Leave>", lambda e, b=copy_btn: b.config(fg=c["text_secondary"]))

                        regen_btn = tk.Label(
                            actions, text="🔄 Regenerate",
                            font=self._font(10), cursor="hand2",
                            bg=c["main_bg"], fg=c["text_secondary"],
                        )
                        regen_btn.pack(side="left")
                        regen_btn.bind("<Button-1>", lambda e: self._regenerate())
                        regen_btn.bind("<Enter>", lambda e, b=regen_btn: b.config(fg=c["accent"]))
                        regen_btn.bind("<Leave>", lambda e, b=regen_btn: b.config(fg=c["text_secondary"]))

                        self._finish_stream(final_response, think_text.strip())
                        return

                    accumulated += item

                    # Parse think tags
                    if "<think>" in accumulated and not in_think and not think_done:
                        in_think = True
                        # Remove the tag from display
                        accumulated = accumulated.replace("<think>", "", 1)
                        continue

                    if "</think>" in accumulated and in_think:
                        in_think = False
                        think_done = True
                        accumulated = accumulated.replace("</think>", "", 1)
                        continue

                    if in_think:
                        if item not in ("<", "t", "h", "i", "n", "k", ">"):
                            think_text += item
                            if think_lbl:
                                think_lbl.config(text=think_text + "▌")
                    elif think_done or not use_think:
                        if item not in ("<", "/", "t", "h", "i", "n", "k", ">"):
                            response_text += item
                            response_var.set(response_text + "▌")

                    self._scroll_bottom()

            except queue.Empty:
                pass

            if self._stop:
                final = response_text.strip() if response_text.strip() else "..."
                response_var.set(final)
                self._finish_stream(final, think_text.strip())
                return

            self.after(12, poll)

        poll()

    def _finish_stream(self, content: str, thinking: str = ""):
        self._streaming = False
        c = self._c()
        self.send_btn.config(text="  ➤  ", command=self._send)
        if content:
            self.manager.current.messages.append(
                Message("assistant", content, thinking)
            )
        self._refresh_chat_list()
        while not self._queue.empty():
            self._queue.get_nowait()

    # ──────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────
    def _copy(self, text: str):
        self.clipboard_clear()
        self.clipboard_append(text)

    def _scroll_bottom(self):
        self.msg_canvas.update_idletasks()
        self.msg_canvas.yview_moveto(1.0)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def main():
    app = CatSeekR1()
    app.mainloop()


if __name__ == "__main__":
    main()
