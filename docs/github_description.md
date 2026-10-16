# GitHub repo text — ready to paste

Everything on this page is copy-paste text for GitHub. Nothing here is code.

---

## 1. About (the short description field)

GitHub's *About* box, top-right of the repo. One sentence, no trailing period,
keyword-rich. Limit ≈ 350 characters — all of these are well under it.

**English (recommended):**

```
Real-time system that tells whether a student is on-task while studying, by fusing a wrist-band IMU, a laptop activity logger and a webcam — then nudges politely and reports focus time
```

**Shorter alternative:**

```
Fuses wrist motion, laptop context and a webcam to detect focus in real time, nudge politely, and report session focus time
```

**بالعربي:**

```
نظام يحدد هل الطالب ما زال مركّزاً أثناء المذاكرة، بدمج ساعة معصم وكاميرا وبرنامج على اللابتوب، ثم يرسل تنبيهاً خفيفاً ويعطي تقرير جلسة
```

---

## 2. Topics (the tag chips)

Paste into the *Topics* field, one at a time (lowercase, hyphens):

```
machine-learning  imu  ble  nrf52840  wearable  attention-detection
on-task  productivity  computer-vision  mediapipe  state-machine
edge-ai  python  sensor-fusion  graduation-project
```

Pick ~10 of the most relevant if you prefer a tidy list.

---

## 3. Social preview / tagline

Used as the one-line pitch above the repo name on social cards and search:

```
AAC — real-time on-task detection for studying: wrist motion + laptop context + webcam, fused into one decision.
```

---

## 4. README intro (top of README.md)

**English:**

```markdown
# AAC — Adaptive Attention Companion

> Real-time on-task detection for students: a wrist band, a laptop and a webcam,
> fused into one attention decision.

Studying is broken by distractions we never notice in the moment — the phone,
mind-wandering, an entertainment tab. At the end of the day, we cannot say how
much of the time was really spent on the task.

**AAC** estimates, in real time, whether a student is still **on-task**,
sends a gentle nudge when distraction persists, and produces a report of focus
time and distractions at the end of the session.

Three sources cover each other's blind spots:

| Source | What it sees | Its blind spot |
|---|---|---|
| **Wrist band (IMU)** | handwriting, page turns, phone handling, walking | cannot see the screen |
| **Laptop logger** | study vs. entertainment, input activity | cannot see paper or the phone |
| **Webcam** | presence, head pose, eye closure | cannot tell writing from zoning out |

The system does **not** measure attention in a neurological sense, is **not** a
medical device, and **does not diagnose**. All processing is local; only counts,
categories and angles are stored — never keystrokes, window titles or video.

**Graduation project.** Currently at phase 0 (data contract).

## Status

- [x] Session data contract & validator
- [ ] Data-collection tools (band firmware, hub, logger, webcam)
- [ ] Pilot dataset with labels
- [ ] Baseline model
- [ ] Sensor fusion & state machine
- [ ] Actions, dashboard and end-to-end demo
```

**بالعربي:**

```markdown
# AAC — الرفيق الذكي للانتباه

> كشف الوقت الحقيقي للتركيز أثناء المذاكرة: ساعة معصم ولابتوب وكاميرا، مدمجون في قرار واحد.

المذاكرة تنقطع بتشتيتات لا نلاحظها في لحظتها — الموبايل، السرحان، فتح تبويب ترفيه.
وفي آخر اليوم لا نعرف كم من الوقت كان مذاكرة فعلاً.

نظام **AAC** يقدّر في الوقت الحقيقي هل الطالب ما زال **مركّزاً (on-task)**، ويرسل
تنبيهاً خفيفاً عند التشتت المستمر، ويعطي في نهاية الجلسة تقريراً بوقت التركيز والتشتيتات.

ثلاثة مصادر تكمّل نقاط العمى عند بعضها:

| المصدر | ما يراه | نقطة عماه |
|---|---|---|
| ساعة المعصم (IMU) | الكتابة، تقليب الصفحات، مسك الموبايل، المشي | لا ترى الشاشة |
| برنامج اللابتوب | دراسة أم ترفيه، نشاط الإدخال | لا يرى الورق ولا الموبايل |
| كاميرا الويب | الحضور، وضعية الرأس، غلق العين | لا تفرّق الكتابة عن السرحان |

النظام **لا** يقيس الانتباه بمعنى عصبي، و**ليس** جهازاً طبياً، و**لا** يشخّص.
كل المعالجة محلية؛ نخزّن أرقاماً فقط (عدّات وفئات وزوايا) — دون حروف، أو عناوين نوافذ، أو فيديو.

**مشروع تخرج.** حالياً في المرحلة صفر (عقد البيانات).
```

---

## 5. If you want a one-paragraph "What is this?" for a blog / form

**English:**

> AAC (Adaptive Attention Companion) is a system that estimates whether a
> student is on-task during a study session. It fuses three weak signals — wrist
> motion from an IMU band, laptop activity and active-window category, and
> coarse engagement cues from a webcam — into a single real-time state. A
> grace-period state machine turns those states into polite nudges instead of
> constant interruptions, and each session ends with a report of focus time and
> distraction events. All processing is local, only aggregate numbers are
> stored, and the system makes no medical claims. The project is a graduation
> project built around a small pilot, a labelled dataset and a lightweight
> classical ML baseline.

**بالعربي:**

> **AAC** نظام يقدّر هل الطالب مركّز أثناء جلسة المذاكرة. يدمج ثلاث إشارات ضعيفة —
> حركة المعصم من ساعة IMU، ونشاط اللابتوب وفئة النافذة النشطة، وإشارات انخراط تقريبية
> من كاميرا الويب — في حالة واحدة آنية. وطبقة state machine بمهلة سماح تحوّل هذه
> الحالات إلى تنبيهات لطيفة بدل مقاطعة مستمرة، وتنتهي كل جلسة بتقرير عن وقت التركيز
> وأحداث التشتت. كل المعالجة محلية، ولا تُخزّن إلا أرقام مجمّعة، والنظام لا يقدّم أي
> ادعاء طبي. المشروع هو مشروع تخرج مبني على pilot صغير ومجموعة بيانات موسومة وbaseline
> بسيط من الـ machine learning الكلاسيكي.
```

---

## 6. Suggested first commit message (if you want a clean start)

```
chore: initial repo — session data contract (schema v1.0.0)

- aacdata: columns, enums, meta models, validator
- schema/meta.schema.json
- docs: schema spec, on-task definition, diagram
- examples: synthetic good + broken sessions
- tests: 8 contract tests
```
