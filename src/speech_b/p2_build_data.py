"""P2 dataset builder: Bhagavad Gita verses.

First attempts free real recitations (Wikimedia Commons / archive.org probes).
Whatever is not obtainable falls back to Edge TTS (hi voices, synthesized
Devanagari shlokas) at multiple voices/rates, labelled SYNTHETIC.
Splits: train = voice Madhur (3 rates), test = held-out voice Swara (+ rates).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OUT / "src"))

from lib_data import (DATA, logged, tts, tts_gtts, to_wav16, fetch, ffmpeg_exe)

AUD = DATA / "gita"
META_PATH = AUD / "p2_dataset.json"
VERSE_TEXT = {
    "1.1": "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥",
    "2.14": "मात्रास्पर्शास्तु कौन्तेय शीतोष्णसुखदुःखदाः। आगमापायिनोऽनित्यास्तांस्तितिक्षस्व भारत॥",
    "2.47": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥",
    "2.62": "ध्यायतो विषयान्पुंसः सङ्गस्तेषूपजायते। सङ्गात्सञ्जायते कामः कामात्क्रोधोऽभिजायते॥",
    "4.7": "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत। अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम्॥",
    "4.8": "परित्राणाय साधूनां विनाशाय च दुष्कृताम्। धर्मसंस्थापनार्थाय सम्भवामि युगे युगे॥",
    "9.22": "अनायाश्चिन्तयन्तो मां ये जनाः पर्युपासते। तेषां नित्याभियुक्तानां योगक्षेमं वहाम्यहम्॥",
    "9.26": "पत्रं पुष्पं फलं तोयं यो मे भक्त्या प्रयच्छति। तदहं भक्त्युपहृतमश्नामि प्रयतात्मनः॥",
    "10.20": "अहमात्मा गुडाकेश सर्वभूताशयस्थितः। अहमादिश्च मध्यं च भूतानामन्त एव च॥",
    "18.66": "सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज। अहं त्वां सर्वपापेभ्यो मोक्षयिष्यामि मा शुचः॥",
}
VOICES_TRAIN = ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"]   # seen in training
VOICES_TEST = ["mr-IN-ManoharNeural", "ne-NP-SagarNeural"]   # fully held-out voices
RATES = ["-10%", "+0%", "+12%"]


def build() -> dict:
    AUD.mkdir(parents=True, exist_ok=True)
    clips = []
    # (optional) probe a couple of known real-recitation candidates
    real = [
        # Wikimedia Commons has some Bhagavad-Gita chanting uploads; probe 2
        "https://upload.wikimedia.org/wikipedia/commons/e/e3/Bhagavad_Gita_chapter_1_part_1.ogg",
        "https://upload.wikimedia.org/wikipedia/commons/8/84/BhagvadGita01.ogg",
    ]
    for i, url in enumerate(real):
        raw = AUD / f"_realprobe{i}.ogg"
        if fetch(url, raw, timeout=25):
            wav = AUD / f"real_probe_{i}.wav"
            if to_wav16(raw, wav, sr=16000):
                logged("web", url, wav, "see Commons page",
                       "[real chant probe; domain sample, excluded from training]")
                clips.append({"path": str(wav), "verse": "unknown", "split": "ignore",
                              "kind": "real-chant-probe"})
        raw.unlink(missing_ok=True)

    for verse, text in VERSE_TEXT.items():
        vid = verse.replace(".", "_")
        for voice in VOICES_TRAIN + VOICES_TEST:
            part = "train" if voice in VOICES_TRAIN else "test"
            voicetag = voice.rsplit("-", 1)[1].replace("Neural", "")
            for rate in RATES:
                tag = f"{vid}_{voicetag}_{rate.replace('%','')}"
                raw = AUD / f"_tts_{tag}.mp3"
                wav = AUD / f"verse_{vid}_{voicetag}_{rate}.wav"
                if wav.exists():
                    clips.append({"path": str(wav), "verse": verse,
                                  "split": part, "kind": "tts",
                                  "voice": voice, "rate": rate})
                    continue
                p = tts(text, raw, voice=voice, rate=rate)
                if p is None:
                    p = tts_gtts(text, raw, lang="hi")
                if p:
                    if to_wav16(p, wav):
                        logged("tts",
                               f"edge-tts voice:{voice} rate:{rate} verse:{verse}",
                               wav, "edge-tts output", "[SYNTHETIC RECITATION]")
                        clips.append({"path": str(wav), "verse": verse,
                                      "split": part, "kind": "tts",
                                      "voice": voice, "rate": rate})
                    p.unlink(missing_ok=True)

    meta = {"clips": clips, "verses": list(VERSE_TEXT),
            "voices": {"train": VOICES_TRAIN, "test": VOICES_TEST},
            "n_clips": len(clips)}
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[p2] {len(clips)} clips over {len(VERSE_TEXT)} verses -> {META_PATH}")
    return meta


if __name__ == "__main__":
    build()
