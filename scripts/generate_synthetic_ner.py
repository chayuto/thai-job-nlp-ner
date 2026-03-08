#!/usr/bin/env python3
import json
import random
import os

SKILLS = [
    "ดูแลผู้สูงอายุ", "ทำอาหาร", "ขับรถ", "ผู้ช่วยพยาบาล", "ดูแลเด็ก", "เฝ้าไข้", 
    "กายภาพบำบัด", "ฉีดยา", "เจาะเลือด", "ทำแผล", "ฟีดอาหาร", "SUCTION", 
    "ดูดเสมหะ", "เปลี่ยนสายยาง", "ยกของ", "งานบ้าน", "แม่บ้าน", "ทำความสะอาด",
    "ซักรีด", "ทำกับข้าว", "ปฐมพยาบาลเบื้องต้น", "ดูแลผู้ป่วยติดเตียง", "ดูแลผู้ป่วย",
    "กายภาพ", "ให้อาหารทางสายยาง", "เขียนโปรแกรม", "ขับรถแบ็คโฮ", "ดูแลผู้ป่วยอัมพฤกษ์",
    "จัดยา", "พาไปหาหมอ", "อาบน้ำเช็ดตัว", "พิมพ์งาน", "คีย์ข้อมูล", "ไลฟ์สด",
    "ขายของออนไลน์", "แอดมินเพจ", "ทำบัญชี", "นวดแผนไทย", "นวดเพื่อสุขภาพ", "ตัดผม"
]
PERSONS = [
    "คุณสมชาย", "พี่แจน", "น้องมิ้น", "ป้าจู", "ลุงชัย", "ยายณี", "ตาบุญ", 
    "พี่นก", "น้องบอย", "หมอเก่ง", "พยาบาลแพม", "คุณแอน", "พี่เอก", "คุณกบ",
    "พี่แจง", "หนูนา", "ตาสมชาย", "ยายมาลี", "น้ากล้วย", "เจ้นิด", "ลุงตู่", "พี่อ้วน"
]
LOCATIONS = [
    "สีลม", "ลาดพร้าว", "รพ.รามาฯ", "รพ.จุฬา", "รพ.ศิริราช", "เชียงใหม่", 
    "สุขุมวิท", "บางนา", "นนทบุรี", "พระราม 2", "รังสิต", "พัทยา", "ภูเก็ต", 
    "ขอนแก่น", "โคราช", "หาดใหญ่", "เอกมัย", "ทองหล่อ", "ดุสิต", "ดอนเมือง",
    "บางกะปิ", "สมุทรปราการ", "ชลบุรี", "รพ.กรุงเทพ", "รพ.พญาไท", "แถวปิ่นเกล้า",
    "แถวฝั่งธน", "ย่านบางนา", "พระราม 9", "บางใหญ่", "พุทธมณฑลสาย 4", "สาย 2"
]
COMPENSATIONS = [
    "18,000 บาท/เดือน", "วันละ 800", "25k-30k", "500/วัน", "15000 บ.", 
    "เดือนละ 20,000", "เรท 1000", "วันละพัน", "12,000 - 15,000", "ชม.ละ 100", 
    "ชั่วโมงละ 150", "20k", "เหมา 5000", "ตามตกลง", "เริ่มต้น 15k", 
    "15,000-18,000", "วันละ 600", "วันละ 700 บาท", "เดือนละ 12k", "1,500 ต่อวัน",
    "รายเดือน 25,000", "เดือนละหมื่นห้า", "คืนละ 800", "คืนละ 1000"
]
TERMS = [
    "เต็มเวลา", "อยู่ประจำ", "กะดึก", "Part-time", "สัญญา 1 ปี", "ไปกลับ", 
    "พาร์ทไทม์", "ทำ จ-ศ", "หยุดเสาร์อาทิตย์", "กะเช้า", "กินอยู่ฟรี", 
    "ประจำ", "ชั่วคราว", "ทำ ส-อา", "freelance", "ทำแค่เสาร์อาทิตย์", 
    "งานประจำ", "ไม่ค้างคืน", "ไป-กลับ", "จันทร์-ศุกร์", "ทำเฉพาะกลางคืน",
    "กะบ่าย", "เข้าเวร 12 ชม.", "เข้าเวร 24 ชม."
]
CONTACTS = [
    "081-234-5678", "Line: @job123", "somchai@gmail.com", "099-999-9999", 
    "โทร 0891234567", "ติดต่อ ib", "inbox มาเลย", "ทักแชท", "Line ไอดี : testcare", 
    "เบอร์ 084-555-5555", "แอดไลน์เบอร์นี้", "inbox ครับ", "โทร 088-888-8888",
    "line: care1234", "โทรมาได้เลยที่ 081-1234567", "ติดต่อ 0922222222", "dm",
    "แอด line 0889999999", "โทร 08x-xxx-xxxx"
]
DEMOGRAPHICS = [
    "อายุ 30-45", "หญิง", "สัญชาติไทย", "เพศชาย", "อายุไม่เกิน 40", 
    "คนไทย", "ผู้หญิง", "ผู้ชาย", "อายุ 20+", "วัยรุ่น", "ป้า", 
    "อายุ 35-50", "สาวๆ", "คนต่างด้าว(มีบัตร)", "อายุไม่เกิน 50 ปี",
    "อายุเกิน 30", "ชาย/หญิง", "วัยกลางคน", "นศ.", "นักศึกษา",
    "คนพื้นที่", "เพศหญิง", "อายุระหว่าง 25-40"
]
EMOJIS = ["🙏", "📞", "💰", "✅", "🔥", "📌", "🏥", "💕", "😊", "✨", "🏥", "👉", "💖", ""]

def template_1():
    s = random.choice(SKILLS)
    loc = random.choice(LOCATIONS)
    comp = random.choice(COMPENSATIONS)
    cont = random.choice(CONTACTS)
    term = random.choice(TERMS)
    e = random.choice(EMOJIS)
    
    raw = f"รับสมัครคน {s} แถว {loc} {e} ทำงานแบบ {term} ให้เงิน {comp} สนใจติดต่อ {cont}"
    entities = [
        {"text": s, "label": "HARD_SKILL"},
        {"text": loc, "label": "LOCATION"},
        {"text": term, "label": "EMPLOYMENT_TERMS"},
        {"text": comp, "label": "COMPENSATION"},
        {"text": cont, "label": "CONTACT"},
    ]
    return raw, entities

def template_2():
    p = random.choice(PERSONS)
    s = random.choice(SKILLS)
    comp = random.choice(COMPENSATIONS)
    cont = random.choice(CONTACTS)
    demo = random.choice(DEMOGRAPHICS)
    
    raw = f"{p} กำลังหาคน {s} คุณสมบัติคือเป็น {demo} ค่ะ เรท {comp} ทักแชท {cont} ด่วนๆๆ"
    entities = [
        {"text": p, "label": "PERSON"},
        {"text": s, "label": "HARD_SKILL"},
        {"text": demo, "label": "DEMOGRAPHIC"},
        {"text": comp, "label": "COMPENSATION"},
        {"text": cont, "label": "CONTACT"},
    ]
    return raw, entities

def template_3():
    loc = random.choice(LOCATIONS)
    cont = random.choice(CONTACTS)
    s1 = random.choice(SKILLS)
    s2 = random.choice(SKILLS[:10])
    demo = random.choice(DEMOGRAPHICS)
    term = random.choice(TERMS)
    e1 = random.choice(EMOJIS)
    e2 = random.choice(EMOJIS)
    
    while s1 == s2: s2 = random.choice(SKILLS)
    
    raw = f"หา{demo} {e1} สามารถ{s1} และ {s2} ได้คับ งาน{term} พิกัด{loc} {e2} {cont}"
    entities = [
        {"text": demo, "label": "DEMOGRAPHIC"},
        {"text": s1, "label": "HARD_SKILL"},
        {"text": s2, "label": "HARD_SKILL"},
        {"text": term, "label": "EMPLOYMENT_TERMS"},
        {"text": loc, "label": "LOCATION"},
        {"text": cont, "label": "CONTACT"},
    ]
    return raw, entities

def template_4():
    p = random.choice(PERSONS)
    loc = random.choice(LOCATIONS)
    comp = random.choice(COMPENSATIONS)
    term = random.choice(TERMS)
    e = random.choice(EMOJIS)
    
    raw = f"{e} ตัวดิฉันชื่อ {p} รับจ้างเฝ้าไข้ {loc} ขอเรทประมาณ {comp} รับงาน{term}นะคะ"
    entities = [
        {"text": p, "label": "PERSON"},
        {"text": loc, "label": "LOCATION"},
        {"text": comp, "label": "COMPENSATION"},
        {"text": term, "label": "EMPLOYMENT_TERMS"},
    ]
    return raw, entities

def template_5():
    s = random.choice(SKILLS)
    loc = random.choice(LOCATIONS)
    comp = random.choice(COMPENSATIONS)
    cont = random.choice(CONTACTS)
    demo = random.choice(DEMOGRAPHICS)
    
    raw = f"ด่วน ‼️ หา{demo} มาดูแลที่ {loc} ต้อง{s} เป็น ให้ {comp} นะคับ -> {cont}"
    entities = [
        {"text": demo, "label": "DEMOGRAPHIC"},
        {"text": loc, "label": "LOCATION"},
        {"text": s, "label": "HARD_SKILL"},
        {"text": comp, "label": "COMPENSATION"},
        {"text": cont, "label": "CONTACT"},
    ]
    return raw, entities

def template_6():
    term = random.choice(TERMS)
    s = random.choice(SKILLS)
    loc = random.choice(LOCATIONS)
    demo = random.choice(DEMOGRAPHICS)
    cont = random.choice(CONTACTS)
    p = random.choice(PERSONS)
    
    raw = f"ตามหา {demo} รับงาน {term} แถว {loc} หน้าที่ {s} ติดต่อ {p} {cont} ด่วนค่ะ"
    entities = [
        {"text": demo, "label": "DEMOGRAPHIC"},
        {"text": term, "label": "EMPLOYMENT_TERMS"},
        {"text": loc, "label": "LOCATION"},
        {"text": s, "label": "HARD_SKILL"},
        {"text": p, "label": "PERSON"},
        {"text": cont, "label": "CONTACT"},
    ]
    return raw, entities

def template_7():
     s1 = random.choice(SKILLS)
     comp = random.choice(COMPENSATIONS)
     cont = random.choice(CONTACTS)
     e = random.choice(EMOJIS)
     raw = f"หาคน {s1} ได้ จ่าย {comp} เลี้ยงข้าวฟรี 1 มื้อ {e} ใครสนใจ {cont}"
     entities = [
        {"text": s1, "label": "HARD_SKILL"},
        {"text": comp, "label": "COMPENSATION"},
        {"text": cont, "label": "CONTACT"}
     ]
     return raw, entities

def template_8():
    demo = random.choice(DEMOGRAPHICS)
    s = random.choice(SKILLS)
    comp = random.choice(COMPENSATIONS)
    term = random.choice(TERMS)
    loc = random.choice(LOCATIONS)
    raw = f"รับสมัคร {demo} งาน {term} สถานที่ {loc} ต้องมี exp. {s} อย่างน้อย 2 ปี ให้เงิน {comp}"
    entities = [
        {"text": demo, "label": "DEMOGRAPHIC"},
        {"text": term, "label": "EMPLOYMENT_TERMS"},
        {"text": loc, "label": "LOCATION"},
        {"text": s, "label": "HARD_SKILL"},
        {"text": comp, "label": "COMPENSATION"}
    ]
    return raw, entities

def template_9():
    p = random.choice(PERSONS)
    s = random.choice(SKILLS)
    term = random.choice(TERMS)
    demo = random.choice(DEMOGRAPHICS)
    cont = random.choice(CONTACTS)
    e = random.choice(EMOJIS)
    raw = f"สวัสดีค่ะ หนูชื่อ {p} เป็น {demo} มีประสบการณ์ {s} หางาน {term} แบบ {term} รบกวน {cont} {e}"
    entities = [
        {"text": p, "label": "PERSON"},
        {"text": demo, "label": "DEMOGRAPHIC"},
        {"text": s, "label": "HARD_SKILL"},
        {"text": term, "label": "EMPLOYMENT_TERMS"},
        {"text": cont, "label": "CONTACT"}
    ]
    return raw, entities

def template_10():
    loc = random.choice(LOCATIONS)
    comp = random.choice(COMPENSATIONS)
    s1 = random.choice(SKILLS)
    s2 = random.choice(SKILLS)
    cont = random.choice(CONTACTS)
    if s1 == s2: s2 = random.choice(SKILLS)
    raw = f"หาคนขับรถ ย่าน{loc} โทร {cont} ถ้า {s1} กับ {s2} ได้จะพิจารณาเป็นพิเศษ \nเรท {comp}"
    entities = [
        {"text": loc, "label": "LOCATION"},
        {"text": cont, "label": "CONTACT"},
        {"text": s1, "label": "HARD_SKILL"},
        {"text": s2, "label": "HARD_SKILL"},
        {"text": comp, "label": "COMPENSATION"}
    ]
    return raw, entities

def template_11():
    p = random.choice(PERSONS)
    loc = random.choice(LOCATIONS)
    comp = random.choice(COMPENSATIONS)
    s = random.choice(SKILLS)
    cont = random.choice(CONTACTS)
    raw = f"{p} ต้องการคน {s} แถว {loc} ให้ค่าจ้าง {comp} ทัก {cont} ได้เลยครับ"
    return raw, [
        {"text": p, "label": "PERSON"},
        {"text": s, "label": "HARD_SKILL"},
        {"text": loc, "label": "LOCATION"},
        {"text": comp, "label": "COMPENSATION"},
        {"text": cont, "label": "CONTACT"}
    ]

def template_12():
    loc = random.choice(LOCATIONS)
    term = random.choice(TERMS)
    demo = random.choice(DEMOGRAPHICS)
    cont = random.choice(CONTACTS)
    raw = f"รับงาน {term} ค่ะ พิกัด {loc} เป็น {demo} ติดต่อ {cont}"
    return raw, [
        {"text": term, "label": "EMPLOYMENT_TERMS"},
        {"text": loc, "label": "LOCATION"},
        {"text": demo, "label": "DEMOGRAPHIC"},
        {"text": cont, "label": "CONTACT"}
    ]

def template_13():
    s = random.choice(SKILLS)
    comp = random.choice(COMPENSATIONS)
    demo = random.choice(DEMOGRAPHICS)
    p = random.choice(PERSONS)
    raw = f"{demo}ที่ทำ {s} เป็น มีไหมคะ {p} รับสมัครอยู่ จ่าย {comp}"
    return raw, [
        {"text": demo, "label": "DEMOGRAPHIC"},
        {"text": s, "label": "HARD_SKILL"},
        {"text": p, "label": "PERSON"},
        {"text": comp, "label": "COMPENSATION"}
    ]

def template_14():
    term = random.choice(TERMS)
    loc = random.choice(LOCATIONS)
    comp = random.choice(COMPENSATIONS)
    cont = random.choice(CONTACTS)
    s1 = random.choice(SKILLS)
    s2 = random.choice(SKILLS)
    if s1 == s2: s2 = random.choice(SKILLS)
    raw = f"งาน {term} {loc} นะคะ ต้องการคนคล่องๆ {s1} และ {s2} ได้ ให้ {comp} แอดมาที่ {cont}"
    return raw, [
        {"text": term, "label": "EMPLOYMENT_TERMS"},
        {"text": loc, "label": "LOCATION"},
        {"text": s1, "label": "HARD_SKILL"},
        {"text": s2, "label": "HARD_SKILL"},
        {"text": comp, "label": "COMPENSATION"},
        {"text": cont, "label": "CONTACT"}
    ]

def template_15():
    p = random.choice(PERSONS)
    demo = random.choice(DEMOGRAPHICS)
    term = random.choice(TERMS)
    cont = random.choice(CONTACTS)
    raw = f"หนู {p} เองค่ะ เป็น {demo} ว่างรับงาน {term} รบกวนทัก {cont} นะคะคุณพี่"
    return raw, [
        {"text": p, "label": "PERSON"},
        {"text": demo, "label": "DEMOGRAPHIC"},
        {"text": term, "label": "EMPLOYMENT_TERMS"},
        {"text": cont, "label": "CONTACT"}
    ]

def template_16():
    s = random.choice(SKILLS)
    loc = random.choice(LOCATIONS)
    term = random.choice(TERMS)
    p = random.choice(PERSONS)
    demo = random.choice(DEMOGRAPHICS)
    comp = random.choice(COMPENSATIONS)
    e = random.choice(EMOJIS)
    raw = f"{e} {p} หาผู้ที่มีคุณสมบัติ {demo} เพื่อช่วย {s} ทำแบบ {term} พิกัด {loc} ค่าตอบแทน {comp}"
    return raw, [
        {"text": p, "label": "PERSON"},
        {"text": demo, "label": "DEMOGRAPHIC"},
        {"text": s, "label": "HARD_SKILL"},
        {"text": term, "label": "EMPLOYMENT_TERMS"},
        {"text": loc, "label": "LOCATION"},
        {"text": comp, "label": "COMPENSATION"}
    ]
    
def generate_post():
    templates = [
        template_1, template_2, template_3, template_4, template_5,
        template_6, template_7, template_8, template_9, template_10,
        template_11, template_12, template_13, template_14, template_15,
        template_16
    ]
    t = random.choice(templates)
    return t()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=400)
    args = parser.parse_args()

    count = args.count
    posts = []
    
    label_counts = {
        "HARD_SKILL": 0,
        "PERSON": 0,
        "LOCATION": 0,
        "COMPENSATION": 0,
        "EMPLOYMENT_TERMS": 0,
        "CONTACT": 0,
        "DEMOGRAPHIC": 0
    }
    
    entities_dropped = 0
    total_entities = 0

    for i in range(count):
        raw, ents = generate_post()
        valid_ents = []
        for e in ents:
            if not e["text"]: continue
            if e["text"] in raw:
                valid_ents.append(e)
                label_counts[e["label"]] += 1
                total_entities += 1
            else:
                entities_dropped += 1
        
        posts.append({
            "id": f"synthetic_{i:04d}",
            "raw_text": raw,
            "entities": valid_ents
        })
        
    out_path = "data/raw/synthetic.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
        
    print(f"Generated {count} total posts.")
    print(f"Total entities: {total_entities}")
    print("Entity count per label:")
    for k, v in label_counts.items():
        print(f"  {k}: {v}")
    print(f"Entities dropped: {entities_dropped}")
    if count > 0:
        print(f"Average entities per post: {total_entities / count:.1f}")
