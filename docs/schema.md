# AAC session data contract (v1.0.0)

عقد بيانات جلسة التسجيل في مشروع AAC. الملف المصدري الوحيد هو `aacdata/columns.py`؛ هذا المستند يشرحه بالعربية والإنجليزي.

## تخطيط الجلسة

```
data/raw/<session_id>/
├── meta.json     # مطلوب، يصف الجلسة
├── imu.csv       # مطلوب، الساعة (50 هرتز أو أكثر)
├── logger.csv    # اختياري، اللابتوب (1 هرتز)
├── vision.csv    # اختياري، الكاميرا (1 هرتز)
├── phone.csv     # مؤجل (1 هرتز)
└── labels.csv    # اختياري حتى التسمية، الحقيقة الأرضية من 인간
```

`session_id` بالصيغة: `p<participant>_<YYYYMMDD>T<HHMMSS>_<laptop|paper>`، ويجب أن يتطابق مع اسم المجلد.

## الساعات والزمن

ساعة واحدة موحدة: ساعة الحاسوب بالتوقيت العالمي (UTC)، بالمللي ثانية منذ epoq. كل ملف يحمل عمود `ts_ms`. الساعة تحتفظ أيضًا بـ `t_dev_ms` (وقت الجهاز الخاص بها) لتشخيص الانقطاع والانزياح في الساعات.

## الملفات

تفاصيل الأعمدة والأنواع والوحدات في `aacdata/columns.py` و `schema/meta.schema.json`. ملخص سريع:

- **imu.csv**: ts_ms, t_dev_ms, ax/ay/az (g), gx/gy/gz (dps), seq
- **logger.csv**: ts_ms, keys, mouse_moves, mouse_clicks, win_switches, category, time_since_input_s
- **vision.csv**: ts_ms, face_present, head_yaw, head_pitch, eye_closure, conf
- **labels.csv**: start_ms, end_ms, activity, task_state, labeler, notes

قواعد التسمية: `UNCERTAIN` ليست تسمية بشرية، بل تنتج من الـ state machine وقت التشغيل. التسمية تُشتق من الفيديو والتقرير الذاتي، **وليس** من فئة النافذة في اللابتوب.

## فحص الجودة (Quality Check)

```bash
python scripts/quality_check.py data/raw/p01_20261020T141530_laptop --require-labels
python scripts/quality_check.py data/raw --json --strict
```

يتحقق السكربت من: عدد الصفوف، المدة، التردد المقاس، الفجوات (> 100 ms)، القيم الناقصة، التوقيتات المكررة، مدى القيم، علامات المزامنة، وصحة ملف التسميات. كود الخروج `1` عند وجود خطأ.
