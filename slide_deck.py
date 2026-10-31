"""Roman Urdu slide deck with a fade-in on each group of shapes."""
from __future__ import annotations

from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

INK = RGBColor(0x17, 0x21, 0x1C)
GREEN = RGBColor(0x1C, 0x46, 0x36)
AMBER = RGBColor(0x8A, 0x5A, 0x1E)
CREAM = RGBColor(0xF7, 0xF5, 0xF1)
WHITE = RGBColor(0xFF, 0xFC, 0xF8)
MUTED = RGBColor(0x3E, 0x48, 0x43)
LINE = RGBColor(0xE3, 0xDD, 0xD4)
SOFT = RGBColor(0xE7, 0xEE, 0xE9)
FONT = "Calibri"
SERIF = "Georgia"
SLIDES = 11


def pct(value: float) -> str:
    return f"{100 * float(value):.2f}%"


def add_rect(slide, left, top, width, height, color: RGBColor):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_card(slide, left, top, width, height, fill: RGBColor = WHITE):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = LINE
    shape.line.width = Pt(1)
    try:
        shape.adjustments[0] = 0.08
    except Exception:
        pass
    return shape


def style_run(run, size: int, bold: bool, color: RGBColor, font: str = FONT) -> None:
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font


def add_text(slide, text, left, top, width, height, size=20, bold=False, color=INK, align=PP_ALIGN.LEFT, font=FONT):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = Emu(0)
    frame.margin_right = Emu(0)
    frame.margin_top = Emu(0)
    frame.margin_bottom = Emu(0)
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    style_run(run, size, bold, color, font)
    return box


def set_notes(slide, notes: str) -> None:
    slide.notes_slide.notes_text_frame.text = notes


def paint_background(slide, color: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_footer(slide, config: dict, page: int) -> None:
    add_rect(slide, Inches(0.48), Inches(7.08), Inches(12.38), Pt(0.75), LINE)
    add_text(slide, f"{config['studentName']}   ·   {config['rollNumber']}", Inches(0.48), Inches(7.14), Inches(8.4), Inches(0.26), size=11, color=MUTED)
    add_text(slide, f"{page}   /   {SLIDES}", Inches(9.4), Inches(7.14), Inches(3.46), Inches(0.26), size=11, color=MUTED, align=PP_ALIGN.RIGHT)


class SlideMotion:
    def __init__(self, slide):
        self.slide = slide
        self.groups: list[list[int]] = []
        self._mark = 0

    def watch(self):
        self._mark = len(self.slide.shapes)
        return self

    def grab(self):
        shapes = list(self.slide.shapes)[self._mark :]
        self._mark = len(self.slide.shapes)
        if shapes:
            self.groups.append([shape.shape_id for shape in shapes])
        return self

    def commit(self) -> None:
        add_fade_transition(self.slide)
        if self.groups:
            set_slide_timing(self.slide, self.groups)


def add_fade_transition(slide) -> None:
    sld = slide._element
    for child in list(sld):
        if child.tag == qn("p:transition"):
            sld.remove(child)
    transition = etree.Element(qn("p:transition"))
    transition.set("spd", "med")
    transition.set("advClick", "1")
    etree.SubElement(transition, qn("p:fade"))
    sld.find(qn("p:cSld")).addnext(transition)


def _ctn(node_id: int, **attrs: str):
    element = etree.Element(qn("p:cTn"))
    element.set("id", str(node_id))
    for key, value in attrs.items():
        element.set(key, value)
    return element


def _delay(parent, milliseconds: int) -> None:
    conditions = etree.SubElement(parent, qn("p:stCondLst"))
    cond = etree.SubElement(conditions, qn("p:cond"))
    cond.set("delay", str(milliseconds))


def _target(parent, shape_id: int) -> None:
    target = etree.SubElement(parent, qn("p:tgtEl"))
    sp_target = etree.SubElement(target, qn("p:spTgt"))
    sp_target.set("spid", str(shape_id))


def _point(parent, moment: str, value: str) -> None:
    point = etree.SubElement(parent, qn("p:tav"))
    point.set("tm", moment)
    holder = etree.SubElement(point, qn("p:val"))
    text = etree.SubElement(holder, qn("p:strVal"))
    text.set("val", value)


def _behavior(parent, name: str, shape_id: int, node) -> None:
    behavior = etree.SubElement(parent, qn("p:cBhvr"))
    behavior.append(node)
    _target(behavior, shape_id)
    names = etree.SubElement(behavior, qn("p:attrNameLst"))
    attr = etree.SubElement(names, qn("p:attrName"))
    attr.text = name


def _entrance(shape_id: int, delay: int, duration: int, ids) -> etree._Element:
    wrapper = etree.Element(qn("p:par"))
    effect = _ctn(ids(), presetID="10", presetClass="entr", presetSubtype="0", fill="hold", grpId="0", nodeType="withEffect")
    _delay(effect, delay)
    behaviors = etree.SubElement(effect, qn("p:childTnLst"))
    shown = etree.SubElement(behaviors, qn("p:set"))
    shown_time = _ctn(ids(), dur="1", fill="hold")
    _delay(shown_time, 0)
    _behavior(shown, "style.visibility", shape_id, shown_time)
    destination = etree.SubElement(shown, qn("p:to"))
    destination_val = etree.SubElement(destination, qn("p:strVal"))
    destination_val.set("val", "visible")
    fade = etree.SubElement(behaviors, qn("p:animEffect"))
    fade.set("transition", "in")
    fade.set("filter", "fade")
    fade_behavior = etree.SubElement(fade, qn("p:cBhvr"))
    fade_behavior.append(_ctn(ids(), dur=str(duration)))
    _target(fade_behavior, shape_id)
    motion = etree.SubElement(behaviors, qn("p:anim"))
    motion.set("calcmode", "lin")
    motion.set("valueType", "num")
    _behavior(motion, "ppt_y", shape_id, _ctn(ids(), dur=str(duration), decel="100000", fill="hold"))
    points = etree.SubElement(motion, qn("p:tavLst"))
    _point(points, "0", "#ppt_y+.045")
    _point(points, "100000", "#ppt_y")
    wrapper.append(effect)
    return wrapper


def set_slide_timing(slide, groups: list[list[int]], stagger: int = 90, duration: int = 420) -> None:
    node_id = 0

    def ids() -> int:
        nonlocal node_id
        node_id += 1
        return node_id

    main_id = 2
    timing = etree.Element(qn("p:timing"))
    tree = etree.SubElement(timing, qn("p:tnLst"))
    root_par = etree.SubElement(tree, qn("p:par"))
    root = _ctn(1, dur="indefinite", restart="never", nodeType="tmRoot")
    root_children = etree.SubElement(root, qn("p:childTnLst"))
    sequence = etree.SubElement(root_children, qn("p:seq"))
    sequence.set("concurrent", "1")
    sequence.set("nextAc", "seek")
    sequence_time = _ctn(main_id, dur="indefinite", nodeType="mainSeq")
    sequence_children = etree.SubElement(sequence_time, qn("p:childTnLst"))
    click = etree.SubElement(sequence_children, qn("p:par"))
    click_time = _ctn(3, fill="hold")
    conditions = etree.SubElement(click_time, qn("p:stCondLst"))
    hold = etree.SubElement(conditions, qn("p:cond"))
    hold.set("delay", "indefinite")
    begin = etree.SubElement(conditions, qn("p:cond"))
    begin.set("evt", "onBegin")
    begin.set("delay", "0")
    begin_target = etree.SubElement(begin, qn("p:tn"))
    begin_target.set("val", str(main_id))
    inner = etree.SubElement(click_time, qn("p:childTnLst"))
    inner_par = etree.SubElement(inner, qn("p:par"))
    inner_time = _ctn(4, fill="hold")
    _delay(inner_time, 0)
    effects = etree.SubElement(inner_time, qn("p:childTnLst"))
    node_id = 4
    shape_ids: list[int] = []
    for index, group in enumerate(groups):
        delay = index * stagger
        for shape_id in group:
            effects.append(_entrance(shape_id, delay, duration, ids))
            shape_ids.append(shape_id)
    inner_par.append(inner_time)
    click.append(click_time)
    sequence.append(sequence_time)
    for tag, event in (("p:prevCondLst", "onPrev"), ("p:nextCondLst", "onNext")):
        event_list = etree.SubElement(sequence, qn(tag))
        cond = etree.SubElement(event_list, qn("p:cond"))
        cond.set("evt", event)
        cond.set("delay", "0")
        target = etree.SubElement(cond, qn("p:tgtEl"))
        etree.SubElement(target, qn("p:sldTgt"))
    root_par.append(root)
    builds = etree.SubElement(timing, qn("p:bldLst"))
    for shape_id in shape_ids:
        build = etree.SubElement(builds, qn("p:bldP"))
        build.set("spid", str(shape_id))
        build.set("grpId", "0")
        build.set("animBg", "1")
    sld = slide._element
    for child in list(sld):
        if child.tag == qn("p:timing"):
            sld.remove(child)
    sld.append(timing)


def content_slide(prs, kicker: str, title: str, notes: str, page: int, config: dict):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    paint_background(slide, CREAM)
    add_rect(slide, 0, 0, Inches(0.12), prs.slide_height, GREEN)
    motion = SlideMotion(slide)
    motion.watch()
    add_text(slide, kicker, Inches(0.48), Inches(0.22), Inches(12.2), Inches(0.24), size=12, bold=True, color=GREEN)
    add_text(slide, title, Inches(0.46), Inches(0.46), Inches(12.3), Inches(0.58), size=30, bold=True, color=INK, font=SERIF)
    add_rect(slide, Inches(0.48), Inches(1.12), Inches(1.15), Pt(2.25), GREEN)
    motion.grab()
    add_footer(slide, config, page)
    motion.watch()
    set_notes(slide, notes)
    return slide, motion


def info_card(slide, left, top, width, height, index: str, title: str, body: str) -> None:
    add_card(slide, left, top, width, height)
    add_text(slide, index, left + Inches(0.22), top + Inches(0.16), width - Inches(0.4), Inches(0.26), size=12, bold=True, color=GREEN)
    add_text(slide, title, left + Inches(0.22), top + Inches(0.46), width - Inches(0.44), Inches(0.7), size=18, bold=True, color=INK, font=SERIF)
    add_text(slide, body, left + Inches(0.22), top + Inches(1.2), width - Inches(0.44), height - Inches(1.38), size=15, color=MUTED)


def stat_card(slide, left, top, width, height, value: str, label: str) -> None:
    add_card(slide, left, top, width, height)
    add_rect(slide, left, top + Inches(0.16), Inches(0.07), height - Inches(0.32), GREEN)
    add_text(slide, value, left + Inches(0.22), top + Inches(0.14), width - Inches(0.36), Inches(0.52), size=26, bold=True, color=GREEN, font=SERIF)
    add_text(slide, label, left + Inches(0.22), top + Inches(0.7), width - Inches(0.36), Inches(0.4), size=13, color=MUTED)


def link_text(config: dict, key: str) -> str:
    value = str(config.get(key, "")).strip()
    return value if value else "Abhi pending"


def build_presentation(config: dict, metrics: dict, out: Path) -> None:
    test = metrics["test"]
    training = metrics["training"]
    counts = metrics["counts"]
    correct = int(round(test["accuracy"] * counts["test"]["total"]))
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = config["title"]
    prs.core_properties.author = config["studentName"]
    prs.core_properties.subject = config["course"]

    title = prs.slides.add_slide(prs.slide_layouts[6])
    paint_background(title, CREAM)
    add_rect(title, 0, 0, Inches(0.18), prs.slide_height, GREEN)
    motion = SlideMotion(title).watch()
    add_text(title, "DEEP LEARNING  ·  SEMESTER PROJECT", Inches(0.7), Inches(1.45), Inches(11.5), Inches(0.32), size=14, bold=True, color=AMBER)
    add_text(title, "Smartphone audio se\nwater-pipe leakage detection", Inches(0.68), Inches(1.9), Inches(11.6), Inches(1.7), size=36, bold=True, color=INK, font=SERIF)
    motion.grab().watch()
    add_text(title, f"{config['studentName']}   ·   {config['rollNumber']}", Inches(0.7), Inches(4.15), Inches(10), Inches(0.36), size=20, color=INK)
    add_text(title, "Teen classes: leak, no leak, environmental noise", Inches(0.7), Inches(4.55), Inches(10), Inches(0.32), size=16, color=MUTED)
    motion.grab()
    set_notes(title, "Assalamualaikum. Mein Zaid Ahmed, roll SU92-BSCSM-F23-553. Project smartphone audio se pipe leak detect karta hai. Teen labels: leak, no leak, aur environmental noise. Animation khud chalegi, beech mein click mat karna.")
    motion.commit()

    slide, motion = content_slide(prs, "01   MASLA", "Leak der se nazar aata hai", "Pipe ka leak tab tak chalta rehta hai jab tak paani nazar na aaye. Ek second ki recording sasti screening hai. Yeh plumber ya safety system nahi.", 2, config)
    cards = [
        ("01", "Chhupa hua leak", "Pipe ya tap ka leak lambe waqt tak chalta rehta hai, jab tak damage saamne na aaye."),
        ("02", "Sasti recording", "Phone mic se sirf ek second ki clip kaafi hai pehle guess ke liye."),
        ("03", "Pehla sort", "Model teen raaste batata hai: leak, khamosh pipe, ya sirf background noise."),
        ("04", "Course demo", "Yeh educational screening hai. Plumber, utility alarm, ya safety decision nahi."),
    ]
    for index, (num, head, body) in enumerate(cards):
        motion.watch()
        info_card(slide, Inches(0.48 + (index % 2) * 6.35), Inches(1.45 + (index // 2) * 2.55), Inches(6.1), Inches(2.4), num, head, body)
        motion.grab()
    motion.commit()

    slide, motion = content_slide(prs, "02   SAWAL", "Kya awaaz se leak alag ho sakta hai?", "Sawal yeh hai ke compact CNN leak, quiet pipe, aur noise alag kare. Training public field clips par hai. Demo phone ki WAV bhi leta hai, lekin phone mic training set nahi hai.", 3, config)
    lines = [
        ("Sawal", "Kya ek second ki awaaz leak, quiet pipe, aur noise mein baant sakti hai?"),
        ("Training", "Public field recordings, Zenodo 18631450. Yeh kitchen phone nahi hain."),
        ("Demo", "Browser WAV file chalata hai, isi network se. Phone clip domain shift hai."),
    ]
    for index, (head, body) in enumerate(lines):
        motion.watch()
        add_card(slide, Inches(0.48), Inches(1.5 + index * 1.7), Inches(12.35), Inches(1.52))
        add_text(slide, head, Inches(0.78), Inches(1.68 + index * 1.7), Inches(3.2), Inches(0.4), size=20, bold=True, color=GREEN, font=SERIF)
        add_text(slide, body, Inches(4.1), Inches(1.7 + index * 1.7), Inches(8.3), Inches(1.05), size=18, color=INK)
        motion.grab()
    motion.commit()

    slide, motion = content_slide(prs, "03   DATASET", "1000 clips, teen folders", "Zenodo 18631450. Leak 500, no leak 386, noise 114. Split seed 42: 700 train, 150 validation, 150 test. Archive git mein nahi, download_dataset.py laata hai.", 4, config)
    stats = [
        ("500", "leak"),
        ("386", "no leak"),
        ("114", "noise"),
        ("700 / 150 / 150", "train / val / test"),
    ]
    for index, (value, label) in enumerate(stats):
        motion.watch()
        stat_card(slide, Inches(0.48 + index * 3.2), Inches(1.55), Inches(3.02), Inches(1.45), value, label)
        motion.grab()
    motion.watch()
    add_card(slide, Inches(0.48), Inches(3.3), Inches(12.35), Inches(3.4), SOFT)
    add_text(slide, "Dongguan leakage-training site ki field recordings. Environmental noise chhoti class hai. Seed 42, stratified 70 / 15 / 15. Test set training ke dauran nahi dekha gaya.", Inches(0.78), Inches(3.55), Inches(11.7), Inches(1.3), size=18, color=INK)
    add_text(slide, "download_dataset.py teen RAR files lata hai. Sirf ek sample WAV har class ka git mein hai.", Inches(0.78), Inches(5.05), Inches(11.7), Inches(1.1), size=16, color=MUTED)
    motion.grab()
    motion.commit()

    slide, motion = content_slide(prs, "04   FEATURES", "Wave se log-mel spectrogram", "16 kilohertz, ek second, Hann STFT n fft 1024, hop 256, 64 mel bands. Shape 1 by 64 by 59. Wahi filterbank browser use karta hai.", 5, config)
    steps = [
        ("1", "16 kHz", "Mono clip ko 16,000 samples par crop ya pad karo. Duration 1.0 second."),
        ("2", "STFT", "Hann window, n_fft 1024, hop 256. 59 frames, center false."),
        ("3", "64 mel", "Slaney filterbank, log power, clip max se minus 80 dB."),
        ("4", "Same math", "Checkpoint mel filterbank save karta hai. Page wahi matrix use karti hai."),
    ]
    for index, (num, head, body) in enumerate(steps):
        motion.watch()
        info_card(slide, Inches(0.48 + (index % 4) * 3.2), Inches(1.55), Inches(3.02), Inches(4.9), num, head, body)
        motion.grab()
    motion.commit()

    slide, motion = content_slide(prs, "05   MODEL", "Chhota spectrogram CNN", "Char convolution blocks, 98,019 parameters. Dropout 0.35, teen logits. model.py ka build_model architecture wapas banata hai.", 6, config)
    blocks = [("16", "pehla block"), ("32", "doosra"), ("64", "teesra"), ("128", "aakhri + pool")]
    for index, (value, label) in enumerate(blocks):
        motion.watch()
        stat_card(slide, Inches(0.48 + index * 3.2), Inches(1.55), Inches(3.02), Inches(1.45), value, label)
        motion.grab()
    motion.watch()
    add_card(slide, Inches(0.48), Inches(3.25), Inches(12.35), Inches(3.45))
    add_text(slide, "Har block: Conv 3×3, batch norm, ReLU. Beech mein max-pool. Aakhir mein adaptive average pool, dropout 0.35, linear head. Output logits hain, softmax predict.py aur browser mein hota hai.", Inches(0.78), Inches(3.5), Inches(11.7), Inches(1.6), size=18, color=INK)
    add_text(slide, "98,019 parameters. Load: build_model(num_classes=3, pretrained=False) phir state_dict.", Inches(0.78), Inches(5.3), Inches(11.7), Inches(0.9), size=16, color=MUTED)
    motion.grab()
    motion.commit()

    slide, motion = content_slide(prs, "06   TRAINING", "RTX 5070, 23 epochs", "Adam 1e-3, class-weighted loss. Noise ka weight lagbhag 2.92. Best validation accuracy 98 percent. Patience 8. Test uske baad ek dafa.", 7, config)
    train_stats = [
        ("23", "epochs chale"),
        ("98.00%", "best val accuracy"),
        ("32", "batch size"),
        ("1e-3", "Adam learning rate"),
    ]
    for index, (value, label) in enumerate(train_stats):
        motion.watch()
        stat_card(slide, Inches(0.48 + index * 3.2), Inches(1.55), Inches(3.02), Inches(1.5), value, label)
        motion.grab()
    motion.watch()
    add_card(slide, Inches(0.48), Inches(3.3), Inches(12.35), Inches(3.4), SOFT)
    add_text(slide, "Loss class-weighted cross-entropy hai. Train weights: leak 0.667, no leak 0.864, noise 2.917. Augment: time roll, frequency mask, time mask, thori noise.", Inches(0.78), Inches(3.55), Inches(11.7), Inches(1.4), size=18, color=INK)
    add_text(slide, "Early stopping validation accuracy dekhti hai. Best checkpoint reload hota hai, phir test ek hi dafa.", Inches(0.78), Inches(5.15), Inches(11.7), Inches(1.0), size=16, color=MUTED)
    motion.grab()
    motion.commit()

    slide, motion = content_slide(prs, "07   RESULTS", "Test accuracy 94.00%", "141 of 150 sahi. Macro F1 93.13 percent. Sab se zyada confusion leak aur no leak ke beech hai, 3 aur 3. Noise class chhoti hai.", 8, config)
    result_stats = [
        ("94.00%", f"accuracy  {correct}/150"),
        ("93.13%", "macro F1"),
        ("95.36%", "leak F1"),
        ("90.91%", "noise F1"),
    ]
    for index, (value, label) in enumerate(result_stats):
        motion.watch()
        stat_card(slide, Inches(0.48 + index * 3.2), Inches(1.5), Inches(3.02), Inches(1.5), value, label)
        motion.grab()
    motion.watch()
    add_card(slide, Inches(0.48), Inches(3.25), Inches(12.35), Inches(3.45))
    add_text(slide, "Confusion, rows true: leak 72 sahi, 3 no-leak. no-leak 54 sahi, 3 leak, 1 noise. noise 15 sahi, 1 leak, 1 no-leak.", Inches(0.78), Inches(3.5), Inches(11.7), Inches(1.3), size=18, color=INK)
    add_text(slide, "no leak F1 93.10%. Weighted F1 93.99%. Noise support test mein sirf 17 hai.", Inches(0.78), Inches(5.05), Inches(11.7), Inches(1.0), size=16, color=MUTED)
    motion.grab()
    motion.commit()

    slide, motion = content_slide(prs, "08   FILES", "Evaluator yeh files khole", "PyTorch file weights/model.pth. Browser file model.onnx. Command python predict.py --audio samples/leak.wav. Class order leak, no_leak, environmental_noise.", 9, config)
    files = [
        ("weights/model.pth", "PyTorch checkpoint. state_dict load karo build_model se."),
        ("weights/model.onnx", "Wahi network. Vercel page browser mein chalati hai."),
        ("predict.py", "python predict.py --audio samples/leak.wav"),
        ("metrics.json", "Accuracy, F1, confusion matrix, training history."),
    ]
    for index, (head, body) in enumerate(files):
        motion.watch()
        add_card(slide, Inches(0.48), Inches(1.45 + index * 1.32), Inches(12.35), Inches(1.2))
        add_text(slide, head, Inches(0.74), Inches(1.62 + index * 1.32), Inches(4.4), Inches(0.7), size=18, bold=True, color=GREEN, font=SERIF)
        add_text(slide, body, Inches(5.2), Inches(1.64 + index * 1.32), Inches(7.2), Inches(0.75), size=16, color=INK)
        motion.grab()
    motion.commit()

    slide, motion = content_slide(prs, "09   LIMITS", "Phone mic alag distribution hai", "Training field sensors par hai, kitchen smartphone par nahi. Noise class chhoti hai. Demo ko leak-repair decision mat banana. collect_audio.py se baad mein ghar ki clips add ho sakti hain.", 10, config)
    limits = [
        ("01", "Domain shift", "Field mics kitchen phone nahi. Phone clip par model confidently galat ho sakta hai."),
        ("02", "Chhoti noise class", "Sirf 114 noise clips, test mein 17. Is class ka F1 noisy estimate hai."),
        ("03", "Optional phone", "collect_audio.py 1 second, 16 kHz clips data/raw mein rakhta hai."),
        ("04", "Safety nahi", "Yeh plumbing tool nahi. Hazardous leak ka faisla is se mat lo."),
    ]
    for index, (num, head, body) in enumerate(limits):
        motion.watch()
        info_card(slide, Inches(0.48 + (index % 2) * 6.35), Inches(1.45 + (index // 2) * 2.55), Inches(6.1), Inches(2.4), num, head, body)
        motion.grab()
    motion.commit()

    slide, motion = content_slide(prs, "10   LINKS", "Repo, site, aur walkthrough", "GitHub aur Vercel links form par bhi jayenge. YouTube aur LinkedIn abhi pending hain agar URL khali hai.", 11, config)
    links = [
        ("GitHub", link_text(config, "githubUrl")),
        ("Website", link_text(config, "vercelUrl")),
        ("YouTube", link_text(config, "youtubeUrl")),
        ("LinkedIn", link_text(config, "linkedinUrl")),
    ]
    for index, (head, body) in enumerate(links):
        motion.watch()
        add_card(slide, Inches(0.48), Inches(1.45 + index * 1.32), Inches(12.35), Inches(1.2))
        add_text(slide, head, Inches(0.74), Inches(1.68 + index * 1.32), Inches(2.2), Inches(0.55), size=18, bold=True, color=GREEN, font=SERIF)
        add_text(slide, body, Inches(3.1), Inches(1.68 + index * 1.32), Inches(9.3), Inches(0.6), size=16, color=INK)
        motion.grab()
    set_notes(slide, "GitHub aur site ke links parho. YouTube aur LinkedIn agar pending hon to keh do ke form par aa jayenge. Shukriya.")
    motion.commit()

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(out)
    except PermissionError:
        alt = out.with_name("presentation_roman.pptx")
        prs.save(alt)
        print(f"{out.name} is open, so the new deck was saved as {alt.name}.")
