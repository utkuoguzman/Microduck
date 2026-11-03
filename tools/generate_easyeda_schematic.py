"""Generate an A3 EasyEDA-Style Engineering Schematic for Microduck / XGO-Duck.

Matches the exact visual language of the user's reference ESP32 DevKit V1 schematic:
- Clean A3 white engineering sheet (`2200 x 1460`) with dot grid, double border, and title block.
- Centerpiece (`Section 2`): Full physical 30-pin `U2: ESP32 DEVKIT V1 (ESP-WROOM-32)`
  board diagram (Left 15 pins + Right 15 pins in exact physical order, RF shield can,
  2.4 GHz PCB antenna, AMS1117, CP2102, PWR/GPIO2 LEDs, EN/BOOT buttons, Micro-USB,
  and color-coded net flag chevrons on every active pin + `NC` markers on unused pins).
- Left Column:
  - `1. POWER SUPPLY & BURN-PROOF 5V ISOLATION (2S LiPo 7.4V + Mini360/MP1584 5V 3A Buck)`
    with real battery, PTC polyfuse, SPST switch, electrolytic capacitors, and ground symbols.
  - `3. HIGH-LEVEL BRAIN: RASPBERRY PI ZERO W (v1.1) (512MB RAM — RUNS WHISTLE + NEEDLE 3 + 30 RL POLICIES)`
    with 40-pin J8 header connection pins, 921600-baud UART0 bridge, USB-OTG option, and exact RAM/SD budget.
  - Dark-header `ELECTRICAL SPECIFICATIONS, MEMORY BUDGET & JST / MOLEX BUS MAP` card at bottom-left.
- Right Column:
  - `4. 1 Mbps HALF-DUPLEX SERVO BUS & 15x FEETECH HD-1910M ACTUATOR CHAINS`
    with real schematic resistor network (`R1 1kΩ` series TX + `R2 10kΩ` pull-up to `+3V3`),
    and the 3 daisy-chained servo branches (`P1` Left Leg IDs 10-14, `P2` Right Leg IDs 20-24,
    `P3` Neck/Head/Beak IDs 30-34).
  - `5. 6-DOF TRUNK IMU (I2C) & CACTUS WHISTLE I2S VOICE MIC + PTT BUTTON`
    with `U4: QMI8658A / MPU6050 6-DOF IMU`, `U5: INMP441 I2S Digital Mic`, and
    `SW2: Momentary Push-to-Talk Button (GPIO27)` schematic symbol.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from xml.sax.saxutils import escape

OUT_DIR = Path(__file__).resolve().parent.parent / "diagrams"


def net_flag_left(
    x: float,
    y: float,
    label: str,
    fill: str,
    stroke: str,
    text_color: str = "#1e293b",
    width: float = 112,
    stub: float = 22,
) -> str:
    """Draw a net flag to the LEFT of a pin at (x, y), with a wire stub connecting to x."""
    fx_right = x - stub
    fx_left = fx_right - width
    h = 20
    hh = h / 2
    tip = 8
    pts = f"{fx_left},{y-hh} {fx_right-tip},{y-hh} {fx_right},{y} {fx_right-tip},{y+hh} {fx_left},{y+hh}"
    return f"""
    <g class="net-flag">
      <line x1="{fx_right}" y1="{y}" x2="{x}" y2="{y}" stroke="{stroke}" stroke-width="1.8"/>
      <polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>
      <text x="{fx_left + (width - tip)/2}" y="{y + 3.5}" fill="{text_color}" font-size="9.5" font-weight="bold" text-anchor="middle">{escape(label)}</text>
    </g>"""


def net_flag_right(
    x: float,
    y: float,
    label: str,
    fill: str,
    stroke: str,
    text_color: str = "#1e293b",
    width: float = 112,
    stub: float = 22,
) -> str:
    """Draw a net flag to the RIGHT of a pin at (x, y), with a wire stub connecting to x."""
    fx_left = x + stub
    fx_right = fx_left + width
    h = 20
    hh = h / 2
    tip = 8
    pts = f"{fx_left+tip},{y-hh} {fx_right},{y-hh} {fx_right},{y+hh} {fx_left+tip},{y+hh} {fx_left},{y}"
    return f"""
    <g class="net-flag">
      <line x1="{x}" y1="{y}" x2="{fx_left}" y2="{y}" stroke="{stroke}" stroke-width="1.8"/>
      <polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>
      <text x="{fx_left + tip + (width - tip)/2}" y="{y + 3.5}" fill="{text_color}" font-size="9.5" font-weight="bold" text-anchor="middle">{escape(label)}</text>
    </g>"""


def gnd_symbol(x: float, y: float, label: str = "GND") -> str:
    """Draw a standard IEEE schematic ground symbol with vertical drop from (x, y)."""
    return f"""
    <g class="sym-gnd">
      <line x1="{x}" y1="{y}" x2="{x}" y2="{y+11}" stroke="#334155" stroke-width="1.8"/>
      <line x1="{x-10}" y1="{y+11}" x2="{x+10}" y2="{y+11}" stroke="#0f172a" stroke-width="2.2"/>
      <line x1="{x-6.5}" y1="{y+15}" x2="{x+6.5}" y2="{y+15}" stroke="#0f172a" stroke-width="1.8"/>
      <line x1="{x-3}" y1="{y+19}" x2="{x+3}" y2="{y+19}" stroke="#0f172a" stroke-width="1.5"/>
      <text x="{x}" y="{y+29}" fill="#475569" font-size="8.5" font-weight="bold" text-anchor="middle">{escape(label)}</text>
    </g>"""


def section_box(x: float, y: float, w: float, h: float, title: str) -> str:
    """Draw a solid white engineering section container with dashed border and top-left pill header."""
    pill_w = max(250, int(len(title) * 6.8 + 26))
    return f"""
    <g class="section-container">
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="6,4"/>
      <rect x="{x+14}" y="{y-12}" width="{pill_w}" height="24" rx="3" fill="#ffffff" stroke="#94a3b8" stroke-width="1.2"/>
      <text x="{x+24}" y="{y+4}" fill="#334155" font-size="10" font-weight="bold">{escape(title)}</text>
    </g>"""


def build_svg_schematic() -> str:
    """Build the complete A3 (2200 x 1460) EasyEDA Standard engineering schematic SVG."""
    W, H = 2200, 1460

    # 30-Pin ESP32 DevKit V1 definition (15 Left, 15 Right in exact physical order)
    left_pins = [
        (15, "EN (ENABLE / RST)", None, "", "", ""),
        (14, "VP (GPIO 36 / ADC1_0)", None, "", "", ""),
        (13, "VN (GPIO 39 / ADC1_3)", None, "", "", ""),
        (12, "D34 (GPIO 34 / ADC1_6)", "VBAT_SENSE", "#fef3c7", "#d97706", "#92400e"),
        (11, "D35 (GPIO 35 / ADC1_7)", None, "", "", ""),
        (10, "D32 (GPIO 32 / Touch 9)", None, "", "", ""),
        (9,  "D33 (GPIO 33 / I2S_SD)", "MIC_SD", "#f3e8ff", "#9333ea", "#581c87"),
        (8,  "D25 (GPIO 25 / I2S_SCK)", "MIC_SCK", "#f3e8ff", "#9333ea", "#581c87"),
        (7,  "D26 (GPIO 26 / I2S_WS)", "MIC_WS", "#f3e8ff", "#9333ea", "#581c87"),
        (6,  "D27 (GPIO 27 / Touch 7)", "BTN_PTT", "#fef9c3", "#ca8a04", "#854d0e"),
        (5,  "D14 (GPIO 14 / Touch 6)", None, "", "", ""),
        (4,  "D12 (GPIO 12 / Strapping)", None, "", "", ""),
        (3,  "D13 (GPIO 13 / Touch 4)", None, "", "", ""),
        (2,  "GND (Digital Ground)", "GND_LEFT", "", "", ""),
        (1,  "VIN (+5.1V Regulated In)", "+5V_MAIN", "#fee2e2", "#dc2626", "#991b1b"),
    ]

    right_pins = [
        (15, "D23 (GPIO 23 / MOSI)", None, "", "", ""),
        (14, "D22 (GPIO 22 / I2C_SCL)", "IMU_SCL", "#dcfce7", "#16a34a", "#14532d"),
        (13, "TX0 (GPIO 1 / UART0_TX)", "PI_UART_RX", "#ccfbf1", "#0d9488", "#115e59"),
        (12, "RX0 (GPIO 3 / UART0_RX)", "PI_UART_TX", "#dcfce7", "#16a34a", "#14532d"),
        (11, "D21 (GPIO 21 / I2C_SDA)", "IMU_SDA", "#dcfce7", "#16a34a", "#14532d"),
        (10, "D19 (GPIO 19 / MISO)", None, "", "", ""),
        (9,  "D18 (GPIO 18 / SCK)", None, "", "", ""),
        (8,  "D5 (GPIO 5 / CS)", None, "", "", ""),
        (7,  "TX2 (GPIO 17 / UART2_TX)", "SERVO_TX2", "#e0f2fe", "#0284c7", "#075985"),
        (6,  "RX2 (GPIO 16 / UART2_RX)", "SERVO_RX2", "#e0f2fe", "#0284c7", "#075985"),
        (5,  "D4 (GPIO 4 / Touch 0)", "BUS_DIR_OE", "#e0f2fe", "#0284c7", "#075985"),
        (4,  "D2 (GPIO 2 / Status LED)", "STATUS_LED", "#fef9c3", "#ca8a04", "#854d0e"),
        (3,  "D15 (GPIO 15 / Strapping)", None, "", "", ""),
        (2,  "GND (Digital Ground)", "GND_RIGHT", "", "", ""),
        (1,  "3V3 (Onboard 3.3V Out)", "+3V3", "#ffe4e6", "#e11d48", "#881337"),
    ]

    # Center MCU Board Geometry
    mcu_x = 890
    mcu_y = 115
    mcu_w = 420
    mcu_h = 1200
    pin_y_start = 198
    pin_y_step = 72.0

    mcu_pins_svg = []
    for idx, (pnum, plabel, net, fill, stroke, tcol) in enumerate(left_pins):
        py = pin_y_start + idx * pin_y_step
        badge_fill = (
            "#7c3aed" if "MIC_" in (net or "")
            else "#ca8a04" if net == "BTN_PTT"
            else "#d97706" if net == "VBAT_SENSE"
            else "#0f172a" if net == "GND_LEFT"
            else "#b91c1c" if net == "+5V_MAIN"
            else "#0284c7"
        )
        mcu_pins_svg.append(f"""
        <rect x="{mcu_x - 8}" y="{py - 9}" width="22" height="18" rx="2" fill="{badge_fill}"/>
        <text x="{mcu_x + 3}" y="{py + 3.5}" fill="#ffffff" font-size="9.5" font-weight="bold" text-anchor="middle">{pnum}</text>
        <text x="{mcu_x + 22}" y="{py + 3.5}" fill="{'#0f172a' if net else '#64748b'}" font-size="10" font-weight="{'bold' if net else 'normal'}">{escape(plabel)}</text>
        """)
        if net == "GND_LEFT":
            mcu_pins_svg.append(f"""
            <line x1="{mcu_x - 36}" y1="{py}" x2="{mcu_x - 8}" y2="{py}" stroke="#0f172a" stroke-width="2"/>
            {gnd_symbol(mcu_x - 36, py, "GND")}
            """)
        elif net:
            mcu_pins_svg.append(net_flag_left(mcu_x - 8, py, net, fill, stroke, tcol, width=106))
        else:
            short = plabel.split()[0]
            mcu_pins_svg.append(f"""
            <line x1="{mcu_x - 24}" y1="{py}" x2="{mcu_x - 8}" y2="{py}" stroke="#cbd5e1" stroke-width="1.4"/>
            <line x1="{mcu_x - 27}" y1="{py - 3}" x2="{mcu_x - 21}" y2="{py + 3}" stroke="#94a3b8" stroke-width="1.4"/>
            <line x1="{mcu_x - 27}" y1="{py + 3}" x2="{mcu_x - 21}" y2="{py - 3}" stroke="#94a3b8" stroke-width="1.4"/>
            <text x="{mcu_x - 34}" y="{py + 3.5}" fill="#94a3b8" font-size="9" text-anchor="end">NC ({short})</text>
            """)

    for idx, (pnum, plabel, net, fill, stroke, tcol) in enumerate(right_pins):
        py = pin_y_start + idx * pin_y_step
        rx = mcu_x + mcu_w
        badge_fill = (
            "#15803d" if net in ("IMU_SCL", "IMU_SDA", "PI_UART_TX")
            else "#0d9488" if net == "PI_UART_RX"
            else "#0369a1" if net in ("SERVO_TX2", "SERVO_RX2", "BUS_DIR_OE")
            else "#0f172a" if net == "GND_RIGHT"
            else "#be123c" if net == "+3V3"
            else "#0284c7"
        )
        mcu_pins_svg.append(f"""
        <rect x="{rx - 14}" y="{py - 9}" width="22" height="18" rx="2" fill="{badge_fill}"/>
        <text x="{rx - 3}" y="{py + 3.5}" fill="#ffffff" font-size="9.5" font-weight="bold" text-anchor="middle">{pnum}</text>
        <text x="{rx - 22}" y="{py + 3.5}" fill="{'#0f172a' if net else '#64748b'}" font-size="10" font-weight="{'bold' if net else 'normal'}" text-anchor="end">{escape(plabel)}</text>
        """)
        if net == "GND_RIGHT":
            mcu_pins_svg.append(f"""
            <line x1="{rx + 8}" y1="{py}" x2="{rx + 36}" y2="{py}" stroke="#0f172a" stroke-width="2"/>
            {gnd_symbol(rx + 36, py, "GND")}
            """)
        elif net:
            mcu_pins_svg.append(net_flag_right(rx + 8, py, net, fill, stroke, tcol, width=106))
        else:
            short = plabel.split()[0]
            mcu_pins_svg.append(f"""
            <line x1="{rx + 8}" y1="{py}" x2="{rx + 24}" y2="{py}" stroke="#cbd5e1" stroke-width="1.4"/>
            <line x1="{rx + 21}" y1="{py - 3}" x2="{rx + 27}" y2="{py + 3}" stroke="#94a3b8" stroke-width="1.4"/>
            <line x1="{rx + 21}" y1="{py + 3}" x2="{rx + 27}" y2="{py - 3}" stroke="#94a3b8" stroke-width="1.4"/>
            <text x="{rx + 34}" y="{py + 3.5}" fill="#94a3b8" font-size="9" text-anchor="start">NC ({short})</text>
            """)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="background:#ffffff;font-family:'Segoe UI',Arial,sans-serif;">
  <defs>
    <pattern id="dot-grid" width="20" height="20" patternUnits="userSpaceOnUse">
      <rect width="20" height="20" fill="#ffffff"/>
      <circle cx="10" cy="10" r="0.75" fill="#e2e8f0"/>
    </pattern>
    <pattern id="major-grid" width="100" height="100" patternUnits="userSpaceOnUse">
      <rect width="100" height="100" fill="url(#dot-grid)"/>
      <path d="M 100 0 L 0 0 0 100" fill="none" stroke="#f1f5f9" stroke-width="0.9"/>
    </pattern>
  </defs>

  <!-- Pure White Sheet Background & A3 Engineering Frame -->
  <rect width="{W}" height="{H}" fill="#ffffff"/>
  <rect x="18" y="18" width="{W-36}" height="{H-36}" fill="#ffffff" stroke="#334155" stroke-width="2.5"/>
  <rect x="26" y="26" width="{W-52}" height="{H-52}" fill="#ffffff" stroke="#64748b" stroke-width="1"/>

  <!-- ========================================================================================= -->
  <!-- SECTION 1 (TOP-LEFT): POWER SUPPLY, 2S LIPO, FUSES & BURN-PROOF 5V ISOLATION              -->
  <!-- ========================================================================================= -->
  {section_box(48, 52, 650, 395, "1. POWER SUPPLY & BURN-PROOF 5V ISOLATION (2S LiPo 7.4V + 5V 3A UBEC)")}

  <g transform="translate(48, 52)">
    <!-- BT1: 2S LiPo Battery Symbol -->
    <text x="30" y="38" fill="#0f172a" font-size="10.5" font-weight="bold">BT1: 2S LiPo</text>
    <text x="30" y="52" fill="#475569" font-size="9">850mAh 25C XT30</text>
    <text x="30" y="64" fill="#64748b" font-size="8.5">7.4V Nom / 8.4V Max</text>

    <!-- Battery plates at y=102 -->
    <line x1="36" y1="102" x2="54" y2="102" stroke="#0f172a" stroke-width="2"/>
    <line x1="54" y1="86" x2="54" y2="118" stroke="#0f172a" stroke-width="2.5"/>
    <line x1="62" y1="93" x2="62" y2="111" stroke="#0f172a" stroke-width="4"/>
    <line x1="70" y1="86" x2="70" y2="118" stroke="#0f172a" stroke-width="2.5"/>
    <line x1="78" y1="93" x2="78" y2="111" stroke="#0f172a" stroke-width="4"/>
    <text x="45" y="84" fill="#dc2626" font-size="11" font-weight="bold">+</text>
    <text x="83" y="84" fill="#475569" font-size="11" font-weight="bold">-</text>

    <line x1="78" y1="102" x2="122" y2="102" stroke="#dc2626" stroke-width="2.2"/>

    <!-- F1: Resettable PTC Fuse Symbol -->
    <text x="155" y="70" fill="#15803d" font-size="10" font-weight="bold" text-anchor="middle">F1 (PTC)</text>
    <text x="155" y="83" fill="#16a34a" font-size="8.5" text-anchor="middle">3A x 3 Resettable</text>
    <rect x="122" y="92" width="66" height="20" rx="2" fill="#dcfce7" stroke="#16a34a" stroke-width="1.8"/>
    <line x1="128" y1="108" x2="162" y2="94" stroke="#15803d" stroke-width="1.6"/>
    <line x1="162" y1="94" x2="182" y2="94" stroke="#15803d" stroke-width="1.6"/>

    <!-- Wire F1 -> SW1 -->
    <line x1="188" y1="102" x2="238" y2="102" stroke="#dc2626" stroke-width="2.2"/>

    <!-- SW1: SPST Main Power Switch -->
    <text x="265" y="68" fill="#0f172a" font-size="10" font-weight="bold" text-anchor="middle">SW1</text>
    <text x="265" y="80" fill="#64748b" font-size="8.5" text-anchor="middle">Main Power Switch</text>
    <circle cx="242" cy="102" r="4" fill="#ffffff" stroke="#0f172a" stroke-width="2"/>
    <line x1="245" y1="100" x2="284" y2="84" stroke="#0f172a" stroke-width="2.4"/>
    <circle cx="288" cy="102" r="4" fill="#ffffff" stroke="#0f172a" stroke-width="2"/>
    <text x="242" y="118" fill="#64748b" font-size="8" text-anchor="middle">COM</text>
    <text x="288" y="118" fill="#64748b" font-size="8" text-anchor="middle">NO</text>

    <!-- Switched +7.4V_VBAT Node -->
    <line x1="292" y1="102" x2="345" y2="102" stroke="#dc2626" stroke-width="2.5"/>
    <circle cx="322" cy="102" r="4" fill="#dc2626"/>

    <!-- Branch 1: Direct High-Current +7.4V_SERVO Net Flag -->
    <line x1="322" y1="102" x2="322" y2="48" stroke="#dc2626" stroke-width="2.5"/>
    <line x1="322" y1="48" x2="352" y2="48" stroke="#dc2626" stroke-width="2.5"/>
    <polygon points="352,37 482,37 492,48 482,59 352,59" fill="#fee2e2" stroke="#dc2626" stroke-width="1.6"/>
    <text x="419" y="52" fill="#991b1b" font-size="9.5" font-weight="bold" text-anchor="middle">+7.4V_SERVO (BUS)</text>

    <!-- Burn-Proof Warning Box -->
    <rect x="368" y="72" width="262" height="62" rx="4" fill="#fef2f2" stroke="#ef4444" stroke-width="1.5"/>
    <text x="499" y="90" fill="#991b1b" font-size="9.5" font-weight="bold" text-anchor="middle">100% BURN-PROOF 5V ISOLATION</text>
    <text x="499" y="106" fill="#b91c1c" font-size="9" text-anchor="middle">7.4V-8.4V LiPo NEVER touches ESP32 or Pi Zero W!</text>
    <text x="499" y="121" fill="#b91c1c" font-size="9" text-anchor="middle">Only feeds Feetech Servo Pin 2 &amp; Buck IN+.</text>

    <!-- Branch 2: Down into U1 Step-Down Buck IN+ -->
    <path d="M 322 102 L 322 145 L 155 145 L 155 215 L 185 215" fill="none" stroke="#dc2626" stroke-width="2.2"/>

    <!-- Battery (-) Return Path -->
    <path d="M 36 102 L 24 102 L 24 260 L 185 260" fill="none" stroke="#334155" stroke-width="1.8" stroke-dasharray="5,3"/>
    <rect x="36" y="156" width="96" height="32" rx="3" fill="#f8fafc" stroke="#475569" stroke-width="1.2"/>
    <text x="84" y="170" fill="#0f172a" font-size="8.5" font-weight="bold" text-anchor="middle">BATTERY (-)</text>
    <text x="84" y="182" fill="#dc2626" font-size="7.5" font-weight="bold" text-anchor="middle">COMMON RETURN</text>

    <!-- U1: MINI360 / MP1584EN / TPS566238 5V 3A STEP-DOWN MODULE -->
    <rect x="185" y="175" width="225" height="110" rx="5" fill="#f0f9ff" stroke="#0284c7" stroke-width="2"/>
    <text x="297" y="196" fill="#0369a1" font-size="10.5" font-weight="bold" text-anchor="middle">U1: MINI360 / MP1584 5V 3A BUCK</text>
    <text x="297" y="210" fill="#475569" font-size="8.5" text-anchor="middle">Synchronous Step-Down (Adjust to 5.10V)</text>
    <text x="195" y="228" fill="#0f172a" font-size="9.5" font-weight="bold">IN+ (7.4V)</text>
    <text x="195" y="264" fill="#0f172a" font-size="9.5" font-weight="bold">IN- (BAT-)</text>
    <text x="400" y="228" fill="#0f172a" font-size="9.5" font-weight="bold" text-anchor="end">OUT+ (+5.1V)</text>
    <text x="400" y="264" fill="#0f172a" font-size="9.5" font-weight="bold" text-anchor="end">OUT- (GND)</text>
    <line x1="205" y1="274" x2="390" y2="274" stroke="#64748b" stroke-width="1.2" stroke-dasharray="3,3"/>
    <text x="297" y="246" fill="#0369a1" font-size="8" font-weight="bold" text-anchor="middle">IN- &amp; OUT- share common ground plane</text>

    <!-- Buck OUT+ -> C1 Bulk Cap -> +5V_MAIN Net Flag -->
    <line x1="410" y1="224" x2="525" y2="224" stroke="#dc2626" stroke-width="2.4"/>
    <circle cx="468" cy="224" r="4" fill="#dc2626"/>

    <!-- C1: 1000uF 16V Electrolytic Filter Capacitor Symbol -->
    <text x="468" y="182" fill="#0f172a" font-size="9.5" font-weight="bold" text-anchor="middle">C1</text>
    <text x="468" y="194" fill="#475569" font-size="8.5" text-anchor="middle">1000uF 16V</text>
    <line x1="468" y1="224" x2="468" y2="244" stroke="#0f172a" stroke-width="1.8"/>
    <line x1="454" y1="244" x2="482" y2="244" stroke="#0f172a" stroke-width="2.5"/>
    <path d="M 454 252 Q 468 246 482 252" fill="none" stroke="#0f172a" stroke-width="2.2"/>
    <text x="448" y="241" fill="#dc2626" font-size="10" font-weight="bold">+</text>
    <line x1="468" y1="250" x2="468" y2="274" stroke="#0f172a" stroke-width="1.8"/>
    {gnd_symbol(468, 274, "GND")}

    <!-- Buck OUT- Ground -->
    <line x1="410" y1="260" x2="428" y2="260" stroke="#0f172a" stroke-width="1.8"/>
    {gnd_symbol(428, 260, "GND")}

    <!-- +5V_MAIN Net Flag -->
    <polygon points="525,213 625,213 635,224 625,235 525,235" fill="#fee2e2" stroke="#dc2626" stroke-width="1.6"/>
    <text x="577" y="228" fill="#991b1b" font-size="9.5" font-weight="bold" text-anchor="middle">+5V_MAIN</text>

    <!-- Bottom summary box inside Section 1 -->
    <rect x="24" y="318" width="606" height="62" rx="4" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
    <text x="36" y="338" fill="#dc2626" font-size="9.2" font-weight="bold">• High-Current +7.4V_SERVO Rail:</text>
    <text x="44" y="354" fill="#334155" font-size="8.8">Feeds 15x Feetech HD-1910M Pin 2 (VCC) via 3x 3A PTC Fuses + 3x 220uF 25V Caps</text>
    <text x="415" y="338" fill="#0284c7" font-size="9.2" font-weight="bold">• Regulated +5V_MAIN Rail:</text>
    <text x="423" y="354" fill="#334155" font-size="8.8">Feeds ESP32 (VIN) &amp; Pi Zero W (Pins 2/4)</text>
    <text x="423" y="368" fill="#15803d" font-size="8.5" font-weight="bold">Zero brownouts during 15-servo flips!</text>
  </g>

  <!-- ========================================================================================= -->
  <!-- SECTION 3 (MIDDLE-LEFT): RASPBERRY PI ZERO W (v1.1) — HIGH-LEVEL BRAIN                     -->
  <!-- ========================================================================================= -->
  {section_box(48, 472, 650, 415, "3. HIGH-LEVEL BRAIN: RASPBERRY PI ZERO W (v1.1) (WHISTLE S2T + NEEDLE 3 + 30 RL ZOO)")}

  <g transform="translate(48, 472)">
    <!-- U3: Raspberry Pi Zero W v1.1 Module Card -->
    <rect x="195" y="30" width="435" height="225" rx="5" fill="#ffffff" stroke="#16a34a" stroke-width="2"/>
    <rect x="195" y="30" width="435" height="30" rx="4" fill="#15803d"/>
    <text x="412" y="49" fill="#ffffff" font-size="11" font-weight="bold" text-anchor="middle">U3: RASPBERRY PI ZERO W (v1.1) — 512MB RAM + 16GB+ MICROSD</text>
    <text x="412" y="75" fill="#166534" font-size="9" font-weight="bold" text-anchor="middle">BCM2835 1GHz ARMv6 · 802.11n Wi-Fi · Runs Whistle S2T + Needle 3 + 30 RL Policies!</text>

    <!-- Left side pins of U3 -->
    <text x="208" y="108" fill="#0f172a" font-size="9.5" font-weight="bold">J8 Pin 2 / 4 (5V_IN)</text>
    {net_flag_left(195, 104, "+5V_MAIN", "#fee2e2", "#dc2626", "#991b1b", width=106)}

    <text x="208" y="144" fill="#0f172a" font-size="9.5" font-weight="bold">J8 Pin 6 / 9 / 14 (GND)</text>
    <line x1="158" y1="140" x2="195" y2="140" stroke="#0f172a" stroke-width="1.8"/>
    {gnd_symbol(158, 140, "GND")}

    <text x="208" y="190" fill="#0f172a" font-size="9.5" font-weight="bold">J8 Pin 8 (GPIO14 / TXD0)</text>
    {net_flag_left(195, 186, "PI_UART_TX", "#dcfce7", "#16a34a", "#14532d", width=106)}

    <text x="208" y="230" fill="#0f172a" font-size="9.5" font-weight="bold">J8 Pin 10 (GPIO15 / RXD0)</text>
    {net_flag_left(195, 226, "PI_UART_RX", "#ccfbf1", "#0d9488", "#115e59", width=106)}

    <!-- Right side of U3: Memory & Storage Budget on Pi Zero W v1.1 -->
    <rect x="372" y="88" width="244" height="154" rx="4" fill="#f0fdf4" stroke="#86efac" stroke-width="1.2"/>
    <text x="494" y="106" fill="#166534" font-size="9.2" font-weight="bold" text-anchor="middle">PI ZERO W v1.1 MEMORY BUDGET (FITS 100%!)</text>
    <text x="382" y="123" fill="#334155" font-size="8.5">• Whistle S2T (`whistle_d4.cact`): 14.5 MB SD / ~32 MB RAM</text>
    <text x="382" y="138" fill="#334155" font-size="8.5">• Needle 3 (`duck_4l.cact`):       15.0 MB SD / ~28 MB RAM</text>
    <text x="382" y="153" fill="#334155" font-size="8.5">• 30 RL Policies (`*.npz`):        11.8 MB SD / ~25 MB RAM</text>
    <text x="382" y="168" fill="#334155" font-size="8.5">• Raspberry Pi OS Lite Headless:   ~45 MB RAM</text>
    <line x1="382" y1="176" x2="606" y2="176" stroke="#86efac" stroke-width="1"/>
    <text x="382" y="191" fill="#15803d" font-size="8.8" font-weight="bold">Total Disk: 41.3 MB / 16,000 MB (0.3% of SD card!)</text>
    <text x="382" y="206" fill="#15803d" font-size="8.8" font-weight="bold">Total RAM : ~130 MB / 512 MB (382 MB RAM Free!)</text>
    <text x="382" y="222" fill="#0369a1" font-size="8.3" font-weight="bold">ESP32 Bridge Firmware: 310 KB / 4 MB Flash (7.5%)</text>
    <text x="382" y="235" fill="#475569" font-size="8">UART0 Link: 921,600 baud (100Hz DQS1 / 50Hz DQC1)</text>

    <!-- Bottom callout inside Section 3 -->
    <rect x="24" y="268" width="606" height="130" rx="4" fill="#f0f9ff" stroke="#38bdf8" stroke-width="1.4"/>
    <text x="36" y="287" fill="#0369a1" font-size="9.5" font-weight="bold">HOW PI ZERO W (v1.1) + ESP32 DEVKIT V1 DIVIDE THE WORKLOAD:</text>
    <text x="36" y="304" fill="#334155" font-size="8.8">1. ESP32 DevKit V1 (4MB Flash): Polls 15x Feetech 1910 servos @ 1Mbps + IMU @ 100Hz, and streams 16kHz I2S mic audio when PTT is held.</text>
    <text x="36" y="320" fill="#334155" font-size="8.8">2. Raspberry Pi Zero W v1.1 (512MB RAM): Runs 50Hz Pure-NumPy RL policy (`pi_zero_client.py`) AND runs Cactus Whistle + Needle 3!</text>
    <text x="36" y="336" fill="#334155" font-size="8.8">3. Why does Whistle + Needle 3 never slow down the 50Hz walking loop? Voice transcription only runs while you hold/release PTT</text>
    <text x="36" y="350" fill="#334155" font-size="8.8">   (when the duck is standing waiting for your command), or can be offloaded to your PC over Wi-Fi UDP HIL (`--hil-port 9528`).</text>
    <text x="36" y="368" fill="#0284c7" font-size="8.8" font-weight="bold">• Zero-Soldering Option: Connect Pi Zero W "USB" OTG port directly to ESP32 Micro-USB (`/dev/ttyUSB0`) for both power &amp; 921600-baud data!</text>
  </g>

  <!-- ========================================================================================= -->
  <!-- BOTTOM-LEFT SPEC CARD: ELECTRICAL SPECIFICATIONS & JST / MOLEX BUS MAP                    -->
  <!-- ========================================================================================= -->
  <g transform="translate(48, 908)">
    <rect width="650" height="510" rx="5" fill="#ffffff" stroke="#334155" stroke-width="1.8"/>
    <rect width="650" height="30" rx="4" fill="#0f172a"/>
    <text x="16" y="20" fill="#ffffff" font-size="10.5" font-weight="bold">ELECTRICAL SPECIFICATIONS, JST / MOLEX SERVO BUS MAP &amp; ASSEMBLY RULES</text>

    <text x="16" y="50" fill="#d97706" font-size="9.5" font-weight="bold">DUAL-HEADER ESP32 ARCHITECTURE (ZERO STRAPPING / ADC / UART CONFLICTS):</text>
    <text x="24" y="66" fill="#334155" font-size="8.7">• LEFT HEADER  = VOICE MIC &amp; POWER: INMP441 I2S Mic (D25 SCK, D26 WS, D33 SD), PTT Button (D27), VBAT ADC1 (D34), +5V_MAIN (VIN).</text>
    <text x="24" y="80" fill="#334155" font-size="8.7">• RIGHT HEADER = ROBOTS &amp; SENSORS: Feetech 1Mbps Bus (D17 TX2, D16 RX2), 6-DOF IMU I2C (D22 SCL, D21 SDA), Pi Zero W UART0 (TX0/RX0).</text>
    <text x="24" y="94" fill="#334155" font-size="8.7">• 100% Clean Boot: Strapping pins GPIO0, GPIO2, GPIO12, GPIO15 are never pulled to invalid boot states during reset.</text>

    <text x="16" y="116" fill="#0284c7" font-size="9.5" font-weight="bold">POLARIZED CONNECTOR PINOUTS (GROUPED BY SUBSYSTEM):</text>
    <text x="24" y="132" fill="#334155" font-size="8.7">• CN1 [BATT_XT30]     (2-Pin): Pin 1 = +7.4V_VBAT (2S LiPo+), Pin 2 = Battery GND (-)</text>
    <text x="24" y="146" fill="#334155" font-size="8.7">• P1  [LEFT_LEG_BUS]  (3-Pin): Pin 1 = FEETECH_DATA_1M, Pin 2 = +7.4V_SERVO (F1 3A), Pin 3 = GND --&gt; Servo IDs 10, 11, 12, 13, 14</text>
    <text x="24" y="160" fill="#334155" font-size="8.7">• P2  [RIGHT_LEG_BUS] (3-Pin): Pin 1 = FEETECH_DATA_1M, Pin 2 = +7.4V_SERVO (F2 3A), Pin 3 = GND --&gt; Servo IDs 20, 21, 22, 23, 24</text>
    <text x="24" y="174" fill="#334155" font-size="8.7">• P3  [HEAD_BEAK_BUS] (3-Pin): Pin 1 = FEETECH_DATA_1M, Pin 2 = +7.4V_SERVO (F3 3A), Pin 3 = GND --&gt; Servo IDs 30, 31, 32, 33, 34</text>
    <text x="24" y="188" fill="#334155" font-size="8.7">• J_IMU [TRUNK_IMU]   (4-Pin): Pin 1 = +3V3, Pin 2 = GND, Pin 3 = IMU_SDA (GPIO21), Pin 4 = IMU_SCL (GPIO22)</text>
    <text x="24" y="202" fill="#334155" font-size="8.7">• J_MIC [WHISTLE_I2S] (5-Pin): Pin 1 = +3V3, Pin 2 = GND (tie L/R), Pin 3 = MIC_SCK (D25), Pin 4 = MIC_WS (D26), Pin 5 = MIC_SD (D33)</text>

    <text x="16" y="224" fill="#16a34a" font-size="9.5" font-weight="bold">CIRCUIT PROTECTION, SERVO ID MAP &amp; DECOUPLING HEADROOM:</text>
    <text x="24" y="240" fill="#334155" font-size="8.7">• Left Leg  (P1): ID 10 (hip_yaw, 0°), ID 11 (hip_roll, -5°), ID 12 (hip_pitch, -22°), ID 13 (knee, 0°), ID 14 (ankle, +22°)</text>
    <text x="24" y="254" fill="#334155" font-size="8.7">• Right Leg (P2): ID 20 (hip_yaw, 0°), ID 21 (hip_roll, +5°), ID 22 (hip_pitch, +22°), ID 23 (knee, 0°), ID 24 (ankle, -22°)</text>
    <text x="24" y="268" fill="#334155" font-size="8.7">• Head/Beak (P3): ID 30 (neck_pitch, +20°), ID 31 (head_pitch, +20°), ID 32 (head_yaw, 0°), ID 33 (head_roll, 0°), ID 34 (mouth, 0°..30°)</text>
    <text x="24" y="282" fill="#334155" font-size="8.7">• Bulk Capacitors C21/C23/C25 (3x 220uF 25V): Placed across +7.4V_SERVO &amp; GND at P1/P2/P3 to absorb 15-servo dynamic torque spikes.</text>
    <text x="24" y="296" fill="#334155" font-size="8.7">• Voltage Divider R4 (30kOhm) / R5 (10kOhm) to GPIO34 (ADC1_6): Scales 8.4V LiPo down to 2.10V so ESP32 monitors battery voltage safely.</text>

    <rect x="16" y="314" width="618" height="180" rx="4" fill="#fffbeb" stroke="#f59e0b" stroke-width="1.4"/>
    <text x="28" y="333" fill="#b45309" font-size="9.5" font-weight="bold">CRITICAL ASSEMBLY &amp; CALIBRATION RULES (READ BEFORE POWER-ON):</text>
    <text x="28" y="350" fill="#78350f" font-size="8.7">1. ADJUST MINI360 / MP1584 FIRST: Before connecting ESP32 or Pi Zero W, turn trimpot until OUT+ measures 5.10V DC on a multimeter.</text>
    <text x="28" y="366" fill="#78350f" font-size="8.7">2. ASSIGN SERVO IDs ONE-BY-ONE: Every new Feetech HD-1910M ships as ID 1. Plug ONE servo at a time into the bus and use the Web UI</text>
    <text x="28" y="380" fill="#78350f" font-size="8.7">   (`http://127.0.0.1:9527` -&gt; Servo Setup) to set IDs 10..14, 20..24, 30..34 and center horn at raw count 2047 before daisy-chaining!</text>
    <text x="28" y="396" fill="#78350f" font-size="8.7">3. CHECK 3-PIN SERVO POLARITY: Pin 1 = SIGNAL (White/Yellow), Pin 2 = +7.4V_SERVO (Red, Center!), Pin 3 = GND (Black/Brown).</text>
    <text x="28" y="412" fill="#78350f" font-size="8.7">4. IMU ORIENTATION: Mount QMI8658A / MPU6050 flat in the trunk base (+X Forward, +Y Left, +Z Up) for accurate 50Hz gravity projection.</text>
    <text x="28" y="428" fill="#78350f" font-size="8.7">5. WHISTLE S2T + NEEDLE 3 ON PI ZERO W: Both `whistle_d4.cact` (14.5MB) and `duck_4l.cact` (15MB) use `.cact` v4 CQ2/CQ4 weights</text>
    <text x="28" y="442" fill="#78350f" font-size="8.7">   and run via `whistle-esp32`&apos;s C/NumPy engine on the Pi Zero W v1.1 (or on your PC over Wi-Fi HIL for &lt;100ms GPU/CPU voice response).</text>
    <text x="28" y="458" fill="#78350f" font-size="8.7">6. OPTIONAL SPEAKER: You can also add a MAX98357A I2S DAC Amp on ESP32 GPIO18 (BCLK), GPIO19 (LRC), GPIO5 (DIN) for quack audio!</text>
  </g>

  <!-- ========================================================================================= -->
  <!-- SECTION 2 (CENTER): MCU — U2: ESP32 DEVKIT V1 (ESP-WROOM-32) FULL 30-PIN PACKAGE           -->
  <!-- ========================================================================================= -->
  {section_box(722, 52, 756, 1366, "2. MCU (LEFT = POWER, MIC & ADC | RIGHT = SERVO BUS, IMU & PI ZERO W UART)")}

  <!-- Top 2.4GHz PCB Antenna Tab -->
  <rect x="{mcu_x + 140}" y="{mcu_y - 26}" width="140" height="26" rx="3" fill="#0f172a" stroke="#334155" stroke-width="1.5"/>
  <path d="M {mcu_x+152} {mcu_y-8} L {mcu_x+152} {mcu_y-20} L {mcu_x+164} {mcu_y-20} L {mcu_x+164} {mcu_y-10} L {mcu_x+176} {mcu_y-10} L {mcu_x+176} {mcu_y-20} L {mcu_x+188} {mcu_y-20} L {mcu_x+188} {mcu_y-8}" fill="none" stroke="#f59e0b" stroke-width="2"/>
  <text x="{mcu_x + 232}" y="{mcu_y - 9}" fill="#fde68a" font-size="7.5" font-weight="bold" text-anchor="middle">2.4GHz ANT</text>

  <!-- Main ESP32 DevKit V1 Body -->
  <rect x="{mcu_x}" y="{mcu_y}" width="{mcu_w}" height="{mcu_h}" rx="6" fill="#ffffff" stroke="#1e293b" stroke-width="2.4"/>
  <rect x="{mcu_x}" y="{mcu_y}" width="{mcu_w}" height="38" rx="5" fill="#0284c7"/>
  <text x="{mcu_x + mcu_w/2}" y="{mcu_y + 24}" fill="#ffffff" font-size="13" font-weight="bold" text-anchor="middle">U2: ESP32 DEVKIT V1 (ESP-WROOM-32 · 30-PIN)</text>

  <!-- Subheaders inside ESP32 -->
  <text x="{mcu_x + 18}" y="{mcu_y + 58}" fill="#0284c7" font-size="8.5" font-weight="bold">LEFT: WHISTLE MIC, PTT &amp; POWER</text>
  <text x="{mcu_x + mcu_w - 18}" y="{mcu_y + 58}" fill="#0284c7" font-size="8.5" font-weight="bold" text-anchor="end">RIGHT: 1Mbps SERVO BUS, IMU &amp; PI UART</text>
  <line x1="{mcu_x + 18}" y1="{mcu_y + 64}" x2="{mcu_x + mcu_w - 18}" y2="{mcu_y + 64}" stroke="#cbd5e1" stroke-width="1"/>

  <!-- Internal ESP-WROOM-32 RF Shield Can -->
  <rect x="{mcu_x + 148}" y="{mcu_y + 92}" width="124" height="145" rx="6" fill="#e2e8f0" stroke="#64748b" stroke-width="1.8"/>
  <text x="{mcu_x + 210}" y="{mcu_y + 122}" fill="#334155" font-size="10" font-weight="bold" text-anchor="middle">ESP-WROOM-32</text>
  <text x="{mcu_x + 210}" y="{mcu_y + 138}" fill="#64748b" font-size="8" text-anchor="middle">FCC ID: 2AC7Z</text>
  <circle cx="{mcu_x + 210}" cy="{mcu_y + 172}" r="16" fill="none" stroke="#64748b" stroke-width="1.4"/>
  <text x="{mcu_x + 210}" y="{mcu_y + 176}" fill="#475569" font-size="8.5" font-weight="bold" text-anchor="middle">Wi-Fi</text>
  <text x="{mcu_x + 210}" y="{mcu_y + 218}" fill="#64748b" font-size="8" text-anchor="middle">4MB Flash · 520KB SRAM</text>

  <!-- All 30 Physical Pins & Net Flags -->
  {''.join(mcu_pins_svg)}

  <!-- Internal LDO (AMS1117), USB-UART (CP2102), LEDs, EN/BOOT Buttons & Micro-USB -->
  <circle cx="{mcu_x + 190}" cy="{mcu_y + 762}" r="4.5" fill="#ef4444"/>
  <text x="{mcu_x + 190}" y="{mcu_y + 777}" fill="#dc2626" font-size="7.5" font-weight="bold" text-anchor="middle">PWR</text>
  <circle cx="{mcu_x + 230}" cy="{mcu_y + 762}" r="4.5" fill="#3b82f6"/>
  <text x="{mcu_x + 230}" y="{mcu_y + 777}" fill="#2563eb" font-size="7.5" font-weight="bold" text-anchor="middle">GPIO2</text>

  <rect x="{mcu_x + 175}" y="{mcu_y + 855}" width="70" height="44" rx="3" fill="#1e293b" stroke="#475569" stroke-width="1.2"/>
  <text x="{mcu_x + 210}" y="{mcu_y + 875}" fill="#e2e8f0" font-size="8.5" font-weight="bold" text-anchor="middle">AMS1117</text>
  <text x="{mcu_x + 210}" y="{mcu_y + 888}" fill="#94a3b8" font-size="7.5" text-anchor="middle">3.3V LDO</text>

  <rect x="{mcu_x + 175}" y="{mcu_y + 930}" width="70" height="58" rx="3" fill="#0f172a" stroke="#475569" stroke-width="1.2"/>
  <text x="{mcu_x + 210}" y="{mcu_y + 956}" fill="#e2e8f0" font-size="8.5" font-weight="bold" text-anchor="middle">CP2102</text>
  <text x="{mcu_x + 210}" y="{mcu_y + 970}" fill="#94a3b8" font-size="7.5" text-anchor="middle">USB-UART</text>

  <!-- Physical Correlation Box at bottom of ESP32 -->
  <rect x="{mcu_x + 26}" y="{mcu_y + 1102}" width="{mcu_w - 52}" height="42" rx="4" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2"/>
  <text x="{mcu_x + mcu_w/2}" y="{mcu_y + 1119}" fill="#334155" font-size="8.8" font-weight="bold" text-anchor="middle">ESP32 DEVKIT V1 (PHYSICAL 30-PIN CORRELATION)</text>
  <text x="{mcu_x + mcu_w/2}" y="{mcu_y + 1134}" fill="#64748b" font-size="7.8" text-anchor="middle">LEFT: Mic, PTT &amp; Power (Pins 1,2,6,7,8,9,12) | RIGHT: Servos, IMU &amp; Pi (Pins 1,2,4,5,6,7,11,12,13,14)</text>

  <!-- EN and BOOT Tactile Buttons -->
  <rect x="{mcu_x + 142}" y="{mcu_y + 1154}" width="28" height="22" rx="3" fill="#ffffff" stroke="#475569" stroke-width="1.5"/>
  <circle cx="{mcu_x + 156}" cy="{mcu_y + 1165}" r="5" fill="#334155"/>
  <text x="{mcu_x + 156}" y="{mcu_y + 1188}" fill="#475569" font-size="8" font-weight="bold" text-anchor="middle">EN</text>

  <rect x="{mcu_x + 250}" y="{mcu_y + 1154}" width="28" height="22" rx="3" fill="#ffffff" stroke="#475569" stroke-width="1.5"/>
  <circle cx="{mcu_x + 264}" cy="{mcu_y + 1165}" r="5" fill="#334155"/>
  <text x="{mcu_x + 264}" y="{mcu_y + 1188}" fill="#475569" font-size="8" font-weight="bold" text-anchor="middle">BOOT</text>

  <!-- Bottom Micro-USB Connector -->
  <rect x="{mcu_x + 176}" y="{mcu_y + mcu_h - 14}" width="68" height="26" rx="3" fill="#e2e8f0" stroke="#475569" stroke-width="1.8"/>
  <text x="{mcu_x + 210}" y="{mcu_y + mcu_h + 3}" fill="#1e293b" font-size="8" font-weight="bold" text-anchor="middle">MICRO-USB</text>

  <!-- ========================================================================================= -->
  <!-- SECTION 4 (TOP-RIGHT): 1 Mbps HALF-DUPLEX TRANSCEIVER & 15x FEETECH HD-1910M SERVO BUS     -->
  <!-- ========================================================================================= -->
  {section_box(1502, 52, 650, 745, "4. 1 Mbps HALF-DUPLEX SERVO BUS & 15x FEETECH HD-1910M ACTUATOR CHAINS")}

  <g transform="translate(1502, 52)">
    <!-- 4A. Half-Duplex 1 Mbps Single-Wire UART Circuit -->
    <rect x="24" y="30" width="602" height="190" rx="5" fill="#f8fafc" stroke="#0284c7" stroke-width="1.6"/>
    <rect x="24" y="30" width="602" height="28" rx="4" fill="#0284c7"/>
    <text x="325" y="48" fill="#ffffff" font-size="10.5" font-weight="bold" text-anchor="middle">HALF-DUPLEX 1,000,000 BAUD TTL BUS TRANSCEIVER (DIY 2-RESISTOR OR LUWU SN74LVC1G126/125)</text>

    <!-- SERVO_TX2 Net Flag -> R1 (1kOhm Series Resistor) -> Node A -->
    {net_flag_left(160, 96, "SERVO_TX2", "#e0f2fe", "#0284c7", "#075985", width=102)}
    <line x1="160" y1="96" x2="210" y2="96" stroke="#0284c7" stroke-width="2"/>
    <rect x="210" y="86" width="58" height="20" fill="#ffffff" stroke="#0f172a" stroke-width="1.8"/>
    <text x="239" y="100" fill="#0f172a" font-size="9" font-weight="bold" text-anchor="middle">R1 1kΩ</text>
    <text x="239" y="80" fill="#64748b" font-size="8" text-anchor="middle">Series TX</text>
    <line x1="268" y1="96" x2="430" y2="96" stroke="#0284c7" stroke-width="2.4"/>

    <!-- SERVO_RX2 Net Flag -> Joins Bus directly at x=315 -->
    {net_flag_left(160, 152, "SERVO_RX2", "#e0f2fe", "#0284c7", "#075985", width=102)}
    <line x1="160" y1="152" x2="315" y2="152" stroke="#0284c7" stroke-width="2"/>
    <line x1="315" y1="152" x2="315" y2="96" stroke="#0284c7" stroke-width="2"/>
    <circle cx="315" cy="96" r="4" fill="#0284c7"/>

    <!-- R2 (10kOhm Pull-Up to +3V3) at x=375 -->
    <circle cx="375" cy="96" r="4" fill="#0284c7"/>
    <line x1="375" y1="96" x2="375" y2="118" stroke="#e11d48" stroke-width="1.8"/>
    <rect x="365" y="118" width="20" height="44" fill="#ffffff" stroke="#0f172a" stroke-width="1.8"/>
    <text x="375" y="144" fill="#0f172a" font-size="8" font-weight="bold" text-anchor="middle">10k</text>
    <line x1="375" y1="162" x2="375" y2="180" stroke="#e11d48" stroke-width="1.8"/>
    <polygon points="345,180 405,180 405,198 345,198" fill="#ffe4e6" stroke="#e11d48" stroke-width="1.4"/>
    <text x="375" y="193" fill="#881337" font-size="8.5" font-weight="bold" text-anchor="middle">+3V3 (R2)</text>

    <!-- Output Net Flag: FEETECH_DATA_1M -->
    <polygon points="430,84 595,84 607,96 595,108 430,108" fill="#bae6fd" stroke="#0284c7" stroke-width="1.8"/>
    <text x="515" y="100" fill="#075985" font-size="9.5" font-weight="bold" text-anchor="middle">FEETECH_DATA_1M</text>
    <text x="515" y="128" fill="#475569" font-size="8.5" text-anchor="middle">Single-Wire Half-Duplex Data Bus</text>
    <text x="515" y="142" fill="#64748b" font-size="8" text-anchor="middle">(Luwu V1.1 PCB uses SN74LVC1G126</text>
    <text x="515" y="155" fill="#64748b" font-size="8" text-anchor="middle">+ SN74LVC1G125 w/ BUS_DIR_OE)</text>

    <!-- 4B. P1: LEFT LEG CHAIN (5x Feetech HD-1910M, IDs 10..14) -->
    <rect x="205" y="242" width="420" height="155" rx="5" fill="#ffffff" stroke="#16a34a" stroke-width="2"/>
    <rect x="205" y="242" width="420" height="28" rx="4" fill="#16a34a"/>
    <text x="415" y="260" fill="#ffffff" font-size="10.5" font-weight="bold" text-anchor="middle">P1: LEFT LEG SERVO DAISY-CHAIN (5x HD-1910M · F1=3A, C21=220uF)</text>

    <text x="218" y="294" fill="#0f172a" font-size="9.5" font-weight="bold">Pin 1: DATA (1Mbps)</text>
    {net_flag_left(205, 290, "FEETECH_DATA_1M", "#bae6fd", "#0284c7", "#075985", width=128)}
    <text x="218" y="328" fill="#0f172a" font-size="9.5" font-weight="bold">Pin 2: VCC (+7.4V)</text>
    {net_flag_left(205, 324, "+7.4V_SERVO", "#fee2e2", "#dc2626", "#991b1b", width=128)}
    <text x="218" y="362" fill="#0f172a" font-size="9.5" font-weight="bold">Pin 3: GND (Common)</text>
    <line x1="168" y1="358" x2="205" y2="358" stroke="#0f172a" stroke-width="1.8"/>
    {gnd_symbol(168, 358, "GND")}

    <line x1="365" y1="276" x2="365" y2="388" stroke="#e2e8f0" stroke-width="1.2"/>
    <text x="378" y="292" fill="#15803d" font-size="9" font-weight="bold">├─► ID 10 : left_hip_yaw    (Home   0.0°)</text>
    <text x="378" y="312" fill="#15803d" font-size="9" font-weight="bold">├─► ID 11 : left_hip_roll   (Home  -5.0°)</text>
    <text x="378" y="332" fill="#15803d" font-size="9" font-weight="bold">├─► ID 12 : left_hip_pitch  (Home -22.0°)</text>
    <text x="378" y="352" fill="#15803d" font-size="9" font-weight="bold">├─► ID 13 : left_knee       (Home   0.0°)</text>
    <text x="378" y="372" fill="#15803d" font-size="9" font-weight="bold">└─► ID 14 : left_ankle      (Home +22.0°)</text>

    <!-- 4C. P2: RIGHT LEG CHAIN (5x Feetech HD-1910M, IDs 20..24) -->
    <rect x="205" y="415" width="420" height="155" rx="5" fill="#ffffff" stroke="#0284c7" stroke-width="2"/>
    <rect x="205" y="415" width="420" height="28" rx="4" fill="#0284c7"/>
    <text x="415" y="433" fill="#ffffff" font-size="10.5" font-weight="bold" text-anchor="middle">P2: RIGHT LEG SERVO DAISY-CHAIN (5x HD-1910M · F2=3A, C23=220uF)</text>

    <text x="218" y="467" fill="#0f172a" font-size="9.5" font-weight="bold">Pin 1: DATA (1Mbps)</text>
    {net_flag_left(205, 463, "FEETECH_DATA_1M", "#bae6fd", "#0284c7", "#075985", width=128)}
    <text x="218" y="501" fill="#0f172a" font-size="9.5" font-weight="bold">Pin 2: VCC (+7.4V)</text>
    {net_flag_left(205, 497, "+7.4V_SERVO", "#fee2e2", "#dc2626", "#991b1b", width=128)}
    <text x="218" y="535" fill="#0f172a" font-size="9.5" font-weight="bold">Pin 3: GND (Common)</text>
    <line x1="168" y1="531" x2="205" y2="531" stroke="#0f172a" stroke-width="1.8"/>
    {gnd_symbol(168, 531, "GND")}

    <line x1="365" y1="449" x2="365" y2="561" stroke="#e2e8f0" stroke-width="1.2"/>
    <text x="378" y="465" fill="#0369a1" font-size="9" font-weight="bold">├─► ID 20 : right_hip_yaw   (Home   0.0°)</text>
    <text x="378" y="485" fill="#0369a1" font-size="9" font-weight="bold">├─► ID 21 : right_hip_roll  (Home  +5.0°)</text>
    <text x="378" y="505" fill="#0369a1" font-size="9" font-weight="bold">├─► ID 22 : right_hip_pitch (Home +22.0°)</text>
    <text x="378" y="525" fill="#0369a1" font-size="9" font-weight="bold">├─► ID 23 : right_knee      (Home   0.0°)</text>
    <text x="378" y="545" fill="#0369a1" font-size="9" font-weight="bold">└─► ID 24 : right_ankle     (Home -22.0°)</text>

    <!-- 4D. P3: HEAD, NECK & BEAK CHAIN (5x Feetech HD-1910M, IDs 30..34) -->
    <rect x="205" y="588" width="420" height="145" rx="5" fill="#ffffff" stroke="#d97706" stroke-width="2"/>
    <rect x="205" y="588" width="420" height="28" rx="4" fill="#d97706"/>
    <text x="415" y="606" fill="#ffffff" font-size="10.5" font-weight="bold" text-anchor="middle">P3: HEAD, NECK &amp; BEAK CHAIN (5x HD-1910M · F3=3A, C25=220uF)</text>

    <text x="218" y="638" fill="#0f172a" font-size="9.5" font-weight="bold">Pin 1: DATA (1Mbps)</text>
    {net_flag_left(205, 634, "FEETECH_DATA_1M", "#bae6fd", "#0284c7", "#075985", width=128)}
    <text x="218" y="670" fill="#0f172a" font-size="9.5" font-weight="bold">Pin 2: VCC (+7.4V)</text>
    {net_flag_left(205, 666, "+7.4V_SERVO", "#fee2e2", "#dc2626", "#991b1b", width=128)}
    <text x="218" y="702" fill="#0f172a" font-size="9.5" font-weight="bold">Pin 3: GND (Common)</text>
    <line x1="168" y1="698" x2="205" y2="698" stroke="#0f172a" stroke-width="1.8"/>
    {gnd_symbol(168, 698, "GND")}

    <line x1="365" y1="620" x2="365" y2="725" stroke="#e2e8f0" stroke-width="1.2"/>
    <text x="378" y="636" fill="#b45309" font-size="9" font-weight="bold">├─► ID 30 : neck_pitch      (Home +20.0°)</text>
    <text x="378" y="655" fill="#b45309" font-size="9" font-weight="bold">├─► ID 31 : head_pitch      (Home +20.0°)</text>
    <text x="378" y="674" fill="#b45309" font-size="9" font-weight="bold">├─► ID 32 : head_yaw        (Home   0.0°)</text>
    <text x="378" y="693" fill="#b45309" font-size="9" font-weight="bold">├─► ID 33 : head_roll       (Home   0.0°)</text>
    <text x="378" y="712" fill="#b45309" font-size="9" font-weight="bold">└─► ID 34 : mouth_beak      (0°..+30° Beak)</text>
  </g>

  <!-- ========================================================================================= -->
  <!-- SECTION 5 (BOTTOM-RIGHT): 6-DOF TRUNK IMU (I2C) & CACTUS WHISTLE I2S MIC + PTT            -->
  <!-- ========================================================================================= -->
  {section_box(1502, 825, 650, 470, "5. 6-DOF TRUNK IMU (I2C) & CACTUS WHISTLE I2S VOICE MIC + PTT BUTTON")}

  <g transform="translate(1502, 825)">
    <!-- U4: QMI8658A / MPU6050 6-DOF IMU -->
    <rect x="205" y="28" width="420" height="150" rx="5" fill="#ffffff" stroke="#16a34a" stroke-width="2"/>
    <rect x="205" y="28" width="420" height="28" rx="4" fill="#16a34a"/>
    <text x="415" y="46" fill="#ffffff" font-size="10.5" font-weight="bold" text-anchor="middle">U4: QMI8658A (0x6B) / MPU6050 (0x68) 6-DOF TRUNK IMU</text>

    <text x="218" y="78" fill="#0f172a" font-size="9.5" font-weight="bold">VCC (3.3V In)</text>
    {net_flag_left(205, 74, "+3V3", "#ffe4e6", "#e11d48", "#881337", width=110)}

    <text x="218" y="106" fill="#0f172a" font-size="9.5" font-weight="bold">GND (Ground)</text>
    <line x1="168" y1="102" x2="205" y2="102" stroke="#0f172a" stroke-width="1.8"/>
    {gnd_symbol(168, 102, "GND")}

    <text x="218" y="138" fill="#0f172a" font-size="9.5" font-weight="bold">SCL (D22 / 400kHz)</text>
    {net_flag_left(205, 134, "IMU_SCL", "#dcfce7", "#16a34a", "#14532d", width=110)}

    <text x="218" y="166" fill="#0f172a" font-size="9.5" font-weight="bold">SDA (D21 / 400kHz)</text>
    {net_flag_left(205, 162, "IMU_SDA", "#dcfce7", "#16a34a", "#14532d", width=110)}

    <rect x="375" y="68" width="232" height="96" rx="4" fill="#f0fdf4" stroke="#86efac" stroke-width="1.2"/>
    <text x="491" y="88" fill="#166534" font-size="9" font-weight="bold" text-anchor="middle">100 Hz Attitude &amp; Gyro</text>
    <text x="491" y="104" fill="#334155" font-size="8.5" text-anchor="middle">• 4.7kΩ pull-ups on SDA &amp; SCL</text>
    <text x="491" y="120" fill="#334155" font-size="8.5" text-anchor="middle">• Decoupling: C1=1uF, C2=100nF</text>
    <text x="491" y="136" fill="#334155" font-size="8.5" text-anchor="middle">• Auto-detected by ESP32 bridge</text>
    <text x="491" y="152" fill="#15803d" font-size="8.2" font-weight="bold" text-anchor="middle">Feeds 50Hz Gravity + Gyro Obs</text>

    <!-- U5: INMP441 I2S Digital Microphone (Cactus Whistle S2T) -->
    <rect x="205" y="196" width="420" height="168" rx="5" fill="#ffffff" stroke="#7c3aed" stroke-width="2"/>
    <rect x="205" y="196" width="420" height="28" rx="4" fill="#7c3aed"/>
    <text x="415" y="214" fill="#ffffff" font-size="10.5" font-weight="bold" text-anchor="middle">U5: INMP441 I2S DIGITAL MICROPHONE (CACTUS WHISTLE S2T)</text>

    <text x="218" y="244" fill="#0f172a" font-size="9.5" font-weight="bold">VDD (3.3V Power)</text>
    {net_flag_left(205, 240, "+3V3", "#ffe4e6", "#e11d48", "#881337", width=110)}

    <text x="218" y="272" fill="#0f172a" font-size="9.5" font-weight="bold">GND &amp; L/R (Left Ch)</text>
    <line x1="168" y1="268" x2="205" y2="268" stroke="#0f172a" stroke-width="1.8"/>
    {gnd_symbol(168, 268, "GND")}

    <text x="218" y="300" fill="#0f172a" font-size="9.5" font-weight="bold">SCK / BCLK (D25)</text>
    {net_flag_left(205, 296, "MIC_SCK", "#f3e8ff", "#9333ea", "#581c87", width=110)}

    <text x="218" y="326" fill="#0f172a" font-size="9.5" font-weight="bold">WS / LRCLK (D26)</text>
    {net_flag_left(205, 322, "MIC_WS", "#f3e8ff", "#9333ea", "#581c87", width=110)}

    <text x="218" y="352" fill="#0f172a" font-size="9.5" font-weight="bold">SD / DOUT (D33)</text>
    {net_flag_left(205, 348, "MIC_SD", "#f3e8ff", "#9333ea", "#581c87", width=110)}

    <rect x="375" y="236" width="232" height="114" rx="4" fill="#faf5ff" stroke="#d8b4fe" stroke-width="1.2"/>
    <text x="491" y="256" fill="#6b21a8" font-size="9" font-weight="bold" text-anchor="middle">Cactus Whistle (`whistle_d4.cact`)</text>
    <text x="491" y="272" fill="#334155" font-size="8.5" text-anchor="middle">• 16 kHz 16-bit Mono I2S Audio</text>
    <text x="491" y="288" fill="#334155" font-size="8.5" text-anchor="middle">• L/R pin tied to GND = Left Ch</text>
    <text x="491" y="304" fill="#334155" font-size="8.5" text-anchor="middle">• Runs on Pi Zero W (14.5MB SD)</text>
    <text x="491" y="320" fill="#334155" font-size="8.5" text-anchor="middle">• Routes transcript to Needle 3</text>
    <text x="491" y="336" fill="#7c3aed" font-size="8.2" font-weight="bold" text-anchor="middle">Fits 100% in Pi Zero W 512MB RAM!</text>

    <!-- SW2: Momentary Push-to-Talk Button (GPIO27) cleanly spaced at y=385..445 -->
    <text x="205" y="388" fill="#0f172a" font-size="9.5" font-weight="bold">SW2: MOMENTARY PUSH-TO-TALK VOICE BUTTON (D27 / INTERNAL PULL-UP)</text>
    {net_flag_left(205, 424, "BTN_PTT", "#fef9c3", "#ca8a04", "#854d0e", width=110)}
    <line x1="205" y1="424" x2="265" y2="424" stroke="#ca8a04" stroke-width="2"/>
    <circle cx="269" cy="424" r="4" fill="#ffffff" stroke="#0f172a" stroke-width="2"/>
    <line x1="262" y1="410" x2="316" y2="410" stroke="#0f172a" stroke-width="2.4"/>
    <line x1="289" y1="410" x2="289" y2="400" stroke="#0f172a" stroke-width="2.4"/>
    <line x1="281" y1="400" x2="297" y2="400" stroke="#0f172a" stroke-width="2.4"/>
    <circle cx="309" cy="424" r="4" fill="#ffffff" stroke="#0f172a" stroke-width="2"/>
    <line x1="313" y1="424" x2="355" y2="424" stroke="#0f172a" stroke-width="1.8"/>
    {gnd_symbol(355, 424, "GND")}
  </g>

  <!-- ========================================================================================= -->
  <!-- ENGINEERING TITLE BLOCK (BOTTOM-RIGHT, MATCHING REFERENCE IMAGE)                          -->
  <!-- ========================================================================================= -->
  <g transform="translate(1502, 1320)">
    <rect width="650" height="98" fill="#ffffff" stroke="#334155" stroke-width="2"/>
    <line x1="0" y1="34" x2="650" y2="34" stroke="#334155" stroke-width="1.5"/>
    <line x1="0" y1="66" x2="650" y2="66" stroke="#cbd5e1" stroke-width="1.2"/>
    <line x1="415" y1="34" x2="415" y2="98" stroke="#334155" stroke-width="1.2"/>
    <line x1="545" y1="66" x2="545" y2="98" stroke="#cbd5e1" stroke-width="1.2"/>

    <text x="14" y="22" fill="#0f172a" font-size="12.5" font-weight="bold">TITLE: MICRODUCK / XGO-DUCK — MAIN SYSTEM SCHEMATIC</text>
    <text x="14" y="54" fill="#475569" font-size="9.5">PROJECT: Microduck Embodied RL Robot (EasyEDA Standard)</text>
    <text x="425" y="54" fill="#0f172a" font-size="9.5" font-weight="bold">SYSTEM: ESP32 DEVKIT V1 + PI ZERO W</text>
    <text x="14" y="85" fill="#475569" font-size="9">ENGINEER: Antigravity Pair-Programmer</text>
    <text x="245" y="85" fill="#64748b" font-size="9">DATE: 2026-11-03</text>
    <text x="425" y="85" fill="#475569" font-size="9">REV: 2.1 (30-PIN)</text>
    <text x="553" y="85" fill="#475569" font-size="9">SHEET: 1/1 (A3)</text>
  </g>
</svg>
"""
    return svg


def build_easyeda_standard_json() -> dict:
    """Build an importable EasyEDA Standard schematic JSON (`docType: 1`)."""
    shapes = [
        "T~L~120~70~0~#0f172a~~20pt~bold~~0~MICRODUCK / XGO-DUCK — MAIN SYSTEM SCHEMATIC (ESP32 DEVKIT V1 + PI ZERO W v1.1 + 15x FEETECH HD-1910M)~1~start~gge10~0~",
        "T~L~120~105~0~#475569~~11pt~~~0~1. Power & Burn-Proof 5V Isolation | 2. 30-Pin ESP32 DevKit V1 | 3. Pi Zero W v1.1 | 4. 1Mbps Feetech Bus (IDs 10-14, 20-24, 30-34) | 5. IMU + INMP441 Whistle S2T~1~start~gge11~0~",
        "R~120~150~~~360~200~#0284c7~2~0~#f0f9ff~gge100~0~",
        "T~L~140~180~0~#0369a1~~12pt~bold~~0~U1: MINI360 / MP1584 5V 3A STEP-DOWN (2S LiPo 7.4V -> +5V_MAIN)~1~start~gge101~0~",
        "T~L~140~215~0~#dc2626~~10pt~bold~~0~BT1: 2S LiPo (7.4V Nom / 8.4V Max) -> F1 (PTC) -> SW1 -> +7.4V_SERVO (Pin 2 ONLY)~1~start~gge102~0~",
        "T~L~140~245~0~#0f172a~~10pt~~~0~U1 OUT+ (+5.1V) + C1 (1000uF 16V) -> +5V_MAIN (Feeds ESP32 VIN & Pi Zero W Pins 2/4)~1~start~gge103~0~",
        "R~120~390~~~360~220~#16a34a~2~0~#f0fdf4~gge200~0~",
        "T~L~140~420~0~#15803d~~12pt~bold~~0~U3: RASPBERRY PI ZERO W (v1.1) (40-PIN J8 HEADER)~1~start~gge201~0~",
        "T~L~140~455~0~#0f172a~~10pt~~~0~J8 Pin 2/4: +5V_MAIN | J8 Pin 6/9/14: GND~1~start~gge202~0~",
        "T~L~140~485~0~#15803d~~10pt~bold~~0~J8 Pin 8 (GPIO14 TXD0) -> PI_UART_TX (to ESP32 RX0 / GPIO3 @ 921600 baud)~1~start~gge203~0~",
        "T~L~140~515~0~#0d9488~~10pt~bold~~0~J8 Pin 10 (GPIO15 RXD0) <- PI_UART_RX (from ESP32 TX0 / GPIO1 @ 921600 baud)~1~start~gge204~0~",
        "R~540~150~~~380~520~#1e293b~2~0~#f8fafc~gge300~0~",
        "T~L~560~180~0~#0284c7~~13pt~bold~~0~U2: ESP32 DEVKIT V1 (ESP-WROOM-32 · 30-PIN)~1~start~gge301~0~",
        "T~L~560~215~0~#0f172a~~10pt~bold~~0~LEFT HEADER (Pins 1..15):~1~start~gge302~0~",
        "T~L~560~240~0~#334155~~9pt~~~0~Pin 1 (VIN) <- +5V_MAIN | Pin 2 (GND) <- GND | Pin 6 (D27) <- BTN_PTT~1~start~gge303~0~",
        "T~L~560~265~0~#7c3aed~~9pt~bold~~0~Pin 7 (D26) -> MIC_WS | Pin 8 (D25) -> MIC_SCK | Pin 9 (D33) <- MIC_SD~1~start~gge304~0~",
        "T~L~560~290~0~#d97706~~9pt~~~0~Pin 12 (D34 / ADC1_6) <- VBAT_SENSE~1~start~gge305~0~",
        "T~L~560~330~0~#0f172a~~10pt~bold~~0~RIGHT HEADER (Pins 1..15):~1~start~gge306~0~",
        "T~L~560~355~0~#e11d48~~9pt~bold~~0~Pin 1 (3V3) -> +3V3 | Pin 2 (GND) <- GND | Pin 4 (D2) -> STATUS_LED~1~start~gge307~0~",
        "T~L~560~380~0~#0284c7~~9pt~bold~~0~Pin 6 (D16 RX2) <- SERVO_RX2 | Pin 7 (D17 TX2) -> SERVO_TX2 (1 Mbps)~1~start~gge308~0~",
        "T~L~560~405~0~#16a34a~~9pt~bold~~0~Pin 11 (D21 SDA) <-> IMU_SDA | Pin 14 (D22 SCL) -> IMU_SCL~1~start~gge309~0~",
        "T~L~560~430~0~#0d9488~~9pt~bold~~0~Pin 12 (RX0) <- PI_UART_TX | Pin 13 (TX0) -> PI_UART_RX~1~start~gge310~0~",
        "R~980~150~~~440~320~#0284c7~2~0~#ffffff~gge400~0~",
        "T~L~1000~180~0~#0284c7~~12pt~bold~~0~4. 1 Mbps HALF-DUPLEX BUS & 15x FEETECH HD-1910M SERVOS~1~start~gge401~0~",
        "T~L~1000~210~0~#0f172a~~10pt~~~0~SERVO_TX2 (D17) --[R1 1kOhm]--+-- FEETECH_DATA_1M (+ R2 10kOhm to +3V3 + SERVO_RX2 D16)~1~start~gge402~0~",
        "T~L~1000~245~0~#16a34a~~10pt~bold~~0~P1 (Left Leg, F1=3A, C21=220uF): IDs 10, 11, 12, 13, 14 (Pin1=DATA, Pin2=+7.4V_SERVO, Pin3=GND)~1~start~gge403~0~",
        "T~L~1000~275~0~#0284c7~~10pt~bold~~0~P2 (Right Leg, F2=3A, C23=220uF): IDs 20, 21, 22, 23, 24 (Pin1=DATA, Pin2=+7.4V_SERVO, Pin3=GND)~1~start~gge404~0~",
        "T~L~1000~305~0~#d97706~~10pt~bold~~0~P3 (Head/Beak, F3=3A, C25=220uF): IDs 30, 31, 32, 33, 34 (Pin1=DATA, Pin2=+7.4V_SERVO, Pin3=GND)~1~start~gge405~0~",
        "R~980~490~~~440~180~#7c3aed~2~0~#faf5ff~gge500~0~",
        "T~L~1000~520~0~#7c3aed~~12pt~bold~~0~5. U4: 6-DOF IMU (I2C) & U5: INMP441 I2S MIC (CACTUS WHISTLE S2T)~1~start~gge501~0~",
        "T~L~1000~550~0~#16a34a~~10pt~~~0~U4 IMU (QMI8658A 0x6B / MPU6050 0x68): VCC=+3V3, GND, SDA=IMU_SDA (D21), SCL=IMU_SCL (D22)~1~start~gge502~0~",
        "T~L~1000~580~0~#7c3aed~~10pt~bold~~0~U5 INMP441 Mic: VDD=+3V3, GND/LR=GND, SCK=MIC_SCK (D25), WS=MIC_WS (D26), SD=MIC_SD (D33)~1~start~gge503~0~",
        "T~L~1000~610~0~#ca8a04~~10pt~bold~~0~SW2 Push-to-Talk Button: BTN_PTT (D27) to GND~1~start~gge504~0~",
    ]
    return {
        "head": {
            "docType": "1",
            "editorVersion": "6.5.40",
            "newgId": True,
            "c_para": {
                "Title": "Microduck / XGO-Duck Main System Schematic (ESP32 DevKit V1 + Pi Zero W v1.1 + 15x Feetech 1910 + Whistle S2T)",
                "Company": "Microduck Embodied RL Suite",
                "Sheet": "1/1",
                "Author": "Antigravity Pair-Programmer",
            },
            "x": "0",
            "y": "0",
        },
        "canvas": "CA~2000~1400~#FFFFFF~yes~#E2E8F0~10~100~100~line~10~pixel~5~0~0",
        "shape": shapes,
        "BBox": {"x": 80, "y": 50, "width": 1400, "height": 680},
    }


def build_interactive_html(svg_content: str, json_content: str) -> str:
    """Wrap the A3 EasyEDA Standard SVG in a zoomable/pannable interactive viewer."""
    escaped_json = json.dumps(json_content)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Microduck / XGO-Duck — EasyEDA Standard A3 Schematic</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; background: #ffffff; color: #0f172a;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      display: flex; flex-direction: column; height: 100vh; overflow: hidden;
    }}
    header {{
      background: #0f172a; color: #f8fafc; padding: 10px 20px;
      display: flex; justify-content: space-between; align-items: center;
      box-shadow: 0 2px 10px rgba(0,0,0,0.2); z-index: 10;
    }}
    header h1 {{ margin: 0; font-size: 15px; color: #fbbf24; }}
    header p {{ margin: 2px 0 0; font-size: 11.5px; color: #cbd5e1; }}
    .toolbar {{ display: flex; gap: 8px; align-items: center; }}
    .btn {{
      background: #1e293b; color: #e2e8f0; border: 1px solid #475569;
      border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: 600;
      cursor: pointer;
    }}
    .btn:hover {{ background: #334155; border-color: #fbbf24; }}
    .btn.primary {{ background: #f59e0b; color: #0f172a; border-color: #f59e0b; font-weight: 700; }}
    #viewport {{
      flex: 1; overflow: auto; display: flex; justify-content: center; align-items: flex-start;
      padding: 18px;
    }}
    #sheet-wrap {{
      background: #ffffff; box-shadow: 0 12px 36px rgba(15, 23, 42, 0.18);
      border-radius: 4px; transform-origin: top center; transition: transform 0.12s ease-out;
      width: 100%; max-width: 2200px;
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>⚡ EasyEDA Standard A3 Schematic — Microduck / XGO-Duck (30-Pin ESP32 DevKit V1 + Pi Zero W v1.1 + 15x Feetech 1910 + Whistle S2T)</h1>
      <p>Full 30-Pin Physical ESP32 Header Correlation · Net Flags · Burn-Proof 5V Isolation · 1 Mbps Half-Duplex Servo Bus · INMP441 I2S Mic</p>
    </div>
    <div class="toolbar">
      <button class="btn" onclick="setZoom(0.75)">75%</button>
      <button class="btn" onclick="setZoom(1.0)">100%</button>
      <button class="btn" onclick="setZoom(1.25)">125%</button>
      <button class="btn" onclick="setZoom(1.5)">150%</button>
      <button class="btn primary" onclick="downloadEasyEdaJson()">⬇ Download EasyEDA (.json)</button>
    </div>
  </header>
  <div id="viewport">
    <div id="sheet-wrap">
      {svg_content}
    </div>
  </div>
  <script>
    const EASYEDA_JSON = {escaped_json};
    function setZoom(z) {{
      document.getElementById('sheet-wrap').style.transform = `scale(${{z}})`;
    }}
    function downloadEasyEdaJson() {{
      const blob = new Blob([EASYEDA_JSON], {{type: 'application/json'}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'Microduck_PiZeroW_ESP32_EasyEDA.json';
      a.click();
      URL.revokeObjectURL(url);
    }}
  </script>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_str = build_svg_schematic()
    eda_dict = build_easyeda_standard_json()
    json_str = json.dumps(eda_dict, indent=2)
    html_str = build_interactive_html(svg_str, json_str)

    svg_path = OUT_DIR / "Microduck_PiZeroW_ESP32_EasyEDA.svg"
    json_path = OUT_DIR / "Microduck_PiZeroW_ESP32_EasyEDA.json"
    html_path = OUT_DIR / "Microduck_EasyEDA_Interactive.html"
    png_path = OUT_DIR / "Microduck_PiZeroW_ESP32_EasyEDA.png"

    svg_path.write_text(svg_str, encoding="utf-8")
    json_path.write_text(json_str, encoding="utf-8")
    html_path.write_text(html_str, encoding="utf-8")

    try:
        import resvg_py

        png_bytes = resvg_py.svg_to_bytes(
            svg_string=svg_str,
            width=2200,
            height=1460,
            font_dirs=["C:/Windows/Fonts"],
        )
        png_path.write_bytes(png_bytes)
        print(f"Generated Crisp PNG    : {png_path} ({len(png_bytes)} bytes)")
    except Exception as exc:
        print(f"PNG render skipped: {exc}")

    print(f"Generated EasyEDA JSON : {json_path} ({len(json_str)} bytes)")
    print(f"Generated Schematic SVG: {svg_path} ({len(svg_str)} bytes)")
    print(f"Generated Interactive  : {html_path} ({len(html_str)} bytes)")


if __name__ == "__main__":
    main()
