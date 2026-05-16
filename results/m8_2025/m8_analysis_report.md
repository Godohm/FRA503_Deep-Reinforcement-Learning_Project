# M8 2025 — Analysis Report: ทำไมผลถึงออกมาติดลบ?

**Dataset:** Train = Jan–Dec 2025 (247 sessions) | Val = Jan–Feb 2026 (40 sessions) | Test = Mar–Apr 2026 (44 sessions)  
**Setup:** 4 algorithms × 3 seeds × 200k steps | HPO best hyperparameters จาก Optuna 50 trials × 50k steps

---

## 1. ภาพรวมผล

| Algorithm | Mean Sharpe | Mean Return | Mean Trades | Transaction Cost (avg) |
|---|---|---|---|---|
| DDQN (linear ε-decay) | -2.810 ± 3.024 | -9.35% | 41 ± 69 | ~$768/run |
| DDQN (exponential ε-decay) | -0.807 ± 1.398 | -0.07% | 1 ± 1 | ~$13/run |
| A2C | -5.351 ± 4.700 | -66.4% | 303 ± 237 | ~$8,120/run |
| PPO | -2.622 ± 0.949 | -29.98% | 148 ± 37 | ~$3,057/run |

ทุก algorithm ได้ Sharpe < 0 บน test split — แต่สาเหตุ **แตกต่างกันอย่างมาก** ในแต่ละ algorithm

---

## 2. DDQN Linear ε-decay — Flat Policy Collapse

### สิ่งที่เกิดขึ้น
2 ใน 3 seeds (seed42, seed2024) เรียนรู้ policy ที่ **ไม่เทรดเลย** (0 trades) หรือเทรดน้อยมาก (2 trades)  
seed123 เทรด 120 ครั้งแต่ได้ win rate เพียง **15%** และขาดทุน -25.47%

### Validation curve ของ seed42

```
step  10k: Sharpe -85.97  trades=4,089  ← ช่วงแรก exploit สุดขีด
step  30k: Sharpe  -2.54  trades=4
step  90k: Sharpe  -1.73  trades=80     ← best checkpoint ณ จุดนั้น
step 200k: Sharpe   0.00  trades=0      ← best final (flat = ดีที่สุด!)
```

### ทำไม DDQN collapse ไปเป็น flat policy?

**กลไก: Cost-Avoidance Learning**

ตอน ε เริ่มลดลง (หลัง step ~70k) Q-network เริ่มตัดสินใจด้วยตัวเอง  
สิ่งที่มันสังเกตได้คือ:
- `action=flat` → reward ≈ 0 (ไม่มี transaction cost)
- `action=long/short` → reward < 0 (ถูกหักค่า spread + commission ทันทีที่เปิด position)

เมื่อ market price เดินแบบ random walk ในระยะสั้น reward เฉลี่ยจาก long/short จึงติดลบ  
Q-values จึงกลายเป็น: **Q(flat) > Q(long) ≈ Q(short)** → agent เลือก flat เสมอ

**หลักฐาน:** seed42 — best checkpoint ถูกเลือกที่ step 200k ซึ่งมี Sharpe=0.00 trades=0  
นั่นหมายความว่า flat policy ชนะ checkpoint ที่ "เทรดจริง" ทุกตัวบน validation

### ทำไม seed123 ถึงเทรดมากกว่า?

seed123 จับได้ checkpoint ที่ step 40k (Sharpe +1.28 บน val, 10 trades)  
แต่ policy ณ step 40k ยังไม่ converge — เมื่อ deploy บน test period ที่ต่างออกไป  
มัน overfit กับ pattern เฉพาะใน val และขาดทุนจนได้ -25.47%

---

## 3. DDQN Exponential ε-decay — Extreme Conservative Collapse

### สิ่งที่เกิดขึ้น
2 ใน 3 seeds: **0 trades, 0 return** (equity คงที่ที่ $10,000)  
seed2024: 2 trades เท่านั้น ขาดทุน -0.22%

### ทำไมถึงแย่กว่า linear ในแง่ความ conservative?

Exponential decay: `ε(t) = 1.0 × (0.05/1.0)^(t/80000)`

การลด ε แบบ exponential ทำให้ agent **สำรวจช้ามากในช่วงแรก** แต่ **หยุดสำรวจเร็วกว่ามาก**  
เมื่อ exploration หมดก่อนที่ Q-values จะแม่นพอ → agent stuck ที่ flat policy เร็วขึ้น

HPO ก็ยืนยันสิ่งนี้: **best trial บน val Sharpe = 0.000** (trial #19)  
นั่นคือ Optuna เองก็พบว่า "flat is best" บน dataset นี้กับ exponential variant

---

## 4. A2C — Overtrade + Transaction Cost Drain

### สิ่งที่เกิดขึ้น
A2C เทรดมากที่สุดและขาดทุนมากที่สุด เฉพาะ seed2024 ขาดทุน -$11,299 จนทุน **ติดลบ** (-$1,299)

### วิเคราะห์ทีละส่วน

| seed | Mark-to-Market PnL | Transaction Cost | Net PnL |
|---|---|---|---|
| 42 | **+$1,174** | -$1,724 | **-$550** |
| 123 | -$242 | -$7,830 | **-$8,072** |
| 2024 | **+$3,508** | -$14,807 | **-$11,299** |

seed42 และ seed2024 มี **directional bet ถูกต้อง** (mtm PnL บวก!) แต่ transaction cost ทำลายกำไรทั้งหมด

### ทำไม A2C overtrade?

**HPO parameter ที่เป็นปัญหา: `n_steps=30`**

n_steps=30 คือ A2C เห็นแค่ **30 bars (= 30 นาที) ต่อ update cycle 1 ครั้ง**

ใน 30 bars แรกของการเปิด position → reward อาจเป็นบวกชั่วคราว  
แต่ A2C ไม่เห็น transaction cost สะสมในภาพ long-term  
มันเรียนรู้ว่า "เปลี่ยน position บ่อยๆ = ดี" เพราะแต่ละ rollout สั้นเกินไปที่จะเห็นผลเสียสะสม

**เปรียบเทียบ:** A2C seed2024 มี 558 trades × ~$26.5/trade ≈ $14,807 cost  
แม้ทายทิศทางถูก $3,508 แต่ cost ทำลายไป **4.2 เท่า**

---

## 5. PPO — Most Consistent แต่ยังขาดทุน

### สิ่งที่เกิดขึ้น
PPO เป็น algorithm ที่ consistent ที่สุด (std Sharpe ±0.949 เทียบกับ A2C ±4.700)  
แต่ทุก seed ยังขาดทุน -20% ถึง -35%

### กรณีน่าสนใจ: PPO seed2024 — Val Sharpe +2.725 vs Test Sharpe -3.039

```
Val performance curve:
  step 10k: Sharpe -52.01  trades=2,419  ← chaos
  step 20k: Sharpe -15.26  trades=777
  step 30k: Sharpe  -0.37  trades=92
  step 40k: Sharpe  +2.73  trades=83   ← BEST CHECKPOINT SAVED
  step 50k: Sharpe  +0.67  trades=85
  ...
  step 200k: Sharpe -2.06  trades=93   ← ลดลงเรื่อยๆ หลัง step 120k
```

Best model ถูก save ที่ step 40k (val Sharpe +2.73)  
แต่เมื่อ deploy บน test period → Sharpe **-3.039**

**สาเหตุ: Val Period Overfitting + Regime Shift**

Val = Jan–Feb 2026 (40 sessions, 2 เดือน) มีขนาดเล็กมาก  
Policy ที่ทำได้ดีใน Jan–Feb 2026 อาจ exploit pattern เฉพาะของช่วงนั้น  
Mar–Apr 2026 มี market regime ต่างออกไป → policy ล้มเหลว

PPO ยังมีปัญหา **exposure time ≈ 1.0** (เกือบ 100% ของเวลาอยู่ใน position)  
ซึ่งหมายถึง policy ไม่เรียนรู้ที่จะ "stay flat" เมื่อ signal ไม่ชัดเจน

---

## 6. สาเหตุหลักที่ผลทุกตัวติดลบ

### 6.1 Transaction Cost สูงเกินกว่า Expected Return

สำหรับ 1 standard lot (100,000 EUR):
- Spread เฉลี่ย ~19 pips = 19 × 1e-5 × 100,000 = **$19 per trade**
- กำไรเฉลี่ยต่อ trade ที่ win rate 50% บน 1-min bar ≈ $5–15

**Transaction cost > Expected directional gain** เกือบทุกกรณี  
Agent ต้องมี win rate และ average win ที่สูงมากพอเพื่อชนะ cost

### 6.2 Train–Test Regime Gap

Train: Jan–Dec 2025 (trend + range + volatility หลายรูปแบบ, 1 ปีเต็ม)  
Test: Mar–Apr 2026 (2 เดือน, regime ที่ agent ไม่เคยเห็น)

Policy ที่ทำได้ดีบน 2025 อาจ overfit กับลักษณะของ data ปีนั้น  
เมื่อ market structure เปลี่ยนใน 2026 → performance drop

### 6.3 Short Validation Window

Val = Jan–Feb 2026 = เพียง **40 sessions** (≈ 2 เดือน)  
Sharpe ratio บน sample เล็กขนาดนี้มี variance สูงมาก  
Best checkpoint เลือกจาก noise ไม่ใช่ true signal → systematic overfitting

### 6.4 Q-Network / Policy Collapse ใน DDQN

ระบบ reward ที่ scale ด้วย net_pnl/initial_equity (O(1e-4)/step)  
ทำให้ Q-value สัญญาณอ่อน → gradient อ่อน → เรียนรู้ช้า  
เมื่อ exploration หมด Q-network ยึดที่ safe action (flat) แทนที่จะ explore profitable actions

---

## 7. สรุปเปรียบเทียบ Algorithm

| | DDQN linear | DDQN exp | A2C | PPO |
|---|---|---|---|---|
| **ปัญหาหลัก** | Flat collapse | Flat collapse (เร็วกว่า) | Overtrade + cost | Val overfitting |
| **Consistency** | ต่ำมาก (variance สูง) | สูง (consistently flat) | ต่ำมาก | สูงที่สุด |
| **Trading frequency** | Bimodal: 0 หรือ 120 trades | เกือบ 0 ทุก seed | สูงมาก (303 avg) | ปานกลาง (148 avg) |
| **Directional accuracy** | Win rate 15% (seed123) | N/A | 28–50% | 40–52% |
| **ความเสี่ยงต่อ regime shift** | ปานกลาง | ต่ำ (ไม่เทรด) | สูงมาก | สูง |
| **แนวทางแก้ไข** | Reward shaping, ลด txn cost | ปรับ decay schedule | เพิ่ม n_steps, penalize overtrade | เพิ่ม val window |

---

## 8. ข้อสังเกตที่ควรทำต่อ (ถ้าต้องการปรับปรุง)

1. **ลด position size**: ใช้ 0.1 lot แทน 1 lot → transaction cost ลด 10 เท่า → threshold ที่ต้องชนะต่ำลง
2. **เพิ่ม val period**: รวม val เป็น 3–4 เดือน เพื่อลด overfitting ต่อ checkpoint selection
3. **A2C: เพิ่ม n_steps**: จาก 30 → 240 (= 4 ชั่วโมง) เพื่อให้ policy เห็น long-term cost
4. **Reward shaping**: เพิ่ม penalty ต่อจำนวน trades สะสม เพื่อสอน agent ให้รู้จัก "stay flat"
5. **Ensemble**: ใช้ majority vote จาก 3 seeds แทนการเลือก single best checkpoint

---

*Generated from M8 2025 experiment results. Test split: Mar–Apr 2026 (44 sessions). All metrics on unseen test data.*
