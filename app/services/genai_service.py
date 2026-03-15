from transformers import pipeline

try:
    generator = pipeline('text-generation', model='HuggingFaceTB/SmolLM-135M', max_new_tokens=50)
except Exception:
    generator = None

# ---------------------------------------------------------------------------
# 36-CLASS FOOD KNOWLEDGE BASE
# Maps each food class → (category, base_score, macro_ratios, micro_profile)
# macro_ratios: (protein_ratio, carb_ratio, fat_ratio, fibre_ratio) relative to calories
# micro_profile: (vitC_factor, vitA_factor, iron_factor, calcium_factor, potassium_base, sodium_base)
# ---------------------------------------------------------------------------
FOOD_KB = {
    # Fruits
    "apple":           ("Fruit",    80, (0.005, 0.14, 0.003, 0.024), (0.08, 0.02, 0.01, 0.005, 107, 1)),
    "banana":          ("Fruit",    72, (0.011, 0.23, 0.003, 0.026), (0.15, 0.01, 0.02, 0.005, 358, 1)),
    "orange":          ("Fruit",    82, (0.009, 0.12, 0.002, 0.024), (0.89, 0.04, 0.01, 0.04, 181, 0)),
    "mango":           ("Fruit",    71, (0.008, 0.15, 0.004, 0.016), (0.44, 0.56, 0.01, 0.01, 168, 1)),
    "watermelon":      ("Fruit",    78, (0.006, 0.08, 0.002, 0.004), (0.13, 0.11, 0.02, 0.007, 112, 1)),
    "grapes":          ("Fruit",    69, (0.007, 0.18, 0.002, 0.009), (0.03, 0.01, 0.02, 0.01, 191, 2)),

    # Vegetables
    "broccoli":        ("Vegetable",90, (0.028, 0.07, 0.004, 0.026), (1.49, 0.12, 0.04, 0.047, 316, 33)),
    "spinach":         ("Vegetable",92, (0.029, 0.036, 0.004, 0.022), (0.47, 1.88, 0.15, 0.099, 558, 79)),
    "carrot":          ("Vegetable",84, (0.009, 0.10, 0.002, 0.028), (0.09, 3.34, 0.02, 0.033, 320, 69)),
    "tomato":          ("Vegetable",81, (0.009, 0.039, 0.002, 0.012), (0.28, 0.17, 0.02, 0.01, 237, 5)),

    # Proteins / Meat / Eggs
    "chicken":         ("Protein",  73, (0.19, 0.00, 0.035, 0.00),   (0.0, 0.01, 0.06, 0.01, 256, 74)),
    "steak":           ("Protein",  62, (0.26, 0.00, 0.12, 0.00),    (0.0, 0.0, 0.14, 0.01, 318, 60)),
    "egg":             ("Protein",  75, (0.126, 0.011, 0.099, 0.00), (0.0, 0.1, 0.06, 0.056, 126, 124)),
    "omelette":        ("Protein",  72, (0.11, 0.01, 0.12, 0.00),    (0.0, 0.12, 0.08, 0.07, 140, 340)),
    "fish":            ("Protein",  79, (0.20, 0.00, 0.05, 0.00),    (0.0, 0.01, 0.05, 0.02, 370, 75)),
    "paneer":          ("Protein",  65, (0.18, 0.04, 0.20, 0.00),    (0.0, 0.04, 0.03, 0.21, 98, 42)),

    # Grains / Breads
    "chapati":         ("Grain",    68, (0.03, 0.18, 0.02, 0.016),   (0.0, 0.0, 0.06, 0.02, 65, 190)),
    "roti":            ("Grain",    68, (0.03, 0.18, 0.02, 0.016),   (0.0, 0.0, 0.06, 0.02, 65, 190)),
    "naan":            ("Grain",    52, (0.04, 0.22, 0.05, 0.008),   (0.0, 0.0, 0.05, 0.03, 90, 400)),
    "puri":            ("Grain",    44, (0.03, 0.22, 0.08, 0.006),   (0.0, 0.0, 0.04, 0.02, 80, 280)),
    "pasta":           ("Grain",    55, (0.05, 0.28, 0.01, 0.013),   (0.0, 0.0, 0.06, 0.01, 45, 6)),
    "waffle":          ("Grain",    38, (0.06, 0.33, 0.10, 0.005),   (0.0, 0.02, 0.08, 0.06, 90, 490)),

    # South Asian / Indian Dishes
    "biryani":         ("Meal",     54, (0.08, 0.25, 0.07, 0.012),   (0.01, 0.02, 0.08, 0.03,  300, 700)),
    "dal":             ("Meal",     76, (0.09, 0.18, 0.03, 0.04),    (0.04, 0.01, 0.14, 0.04, 400, 380)),
    "dosa":            ("Meal",     66, (0.04, 0.20, 0.03, 0.009),   (0.0, 0.0, 0.05, 0.03, 130, 360)),
    "idli":            ("Meal",     72, (0.04, 0.17, 0.005, 0.005),  (0.0, 0.0, 0.04, 0.03, 80, 190)),
    "poha":            ("Meal",     65, (0.03, 0.21, 0.05, 0.006),   (0.0, 0.0, 0.05, 0.01, 120, 250)),
    "upma":            ("Meal",     67, (0.04, 0.22, 0.05, 0.008),   (0.0, 0.0, 0.04, 0.02, 140, 320)),
    "vada":            ("Meal",     49, (0.05, 0.23, 0.11, 0.018),   (0.0, 0.0, 0.05, 0.03, 110, 360)),
    "samosa":          ("Fast Food",40, (0.04, 0.30, 0.16, 0.022),   (0.02, 0.01, 0.05, 0.02, 130, 420)),
    "kofta":           ("Meal",     57, (0.10, 0.10, 0.12, 0.015),   (0.05, 0.03, 0.07, 0.04, 200, 480)),
    "thali":           ("Meal",     70, (0.08, 0.22, 0.06, 0.028),   (0.05, 0.05, 0.10, 0.05, 350, 580)),
    "indian_thali":    ("Meal",     70, (0.08, 0.22, 0.06, 0.028),   (0.05, 0.05, 0.10, 0.05, 350, 580)),
    "fried_rice":      ("Meal",     51, (0.06, 0.26, 0.06, 0.01),    (0.01, 0.01, 0.05, 0.01, 190, 680)),

    # Salads
    "caesar_salad":    ("Salad",    63, (0.04, 0.05, 0.08, 0.018),   (0.15, 0.10, 0.04, 0.08, 220, 680)),
    "salad":           ("Salad",    79, (0.03, 0.06, 0.04, 0.022),   (0.25, 0.30, 0.06, 0.08, 350, 120)),

    # Soups
    "soup":            ("Soup",     61, (0.04, 0.08, 0.02, 0.012),   (0.10, 0.05, 0.05, 0.04, 280, 850)),

    # Sushi
    "sushi":           ("Meal",     74, (0.12, 0.24, 0.02, 0.005),   (0.02, 0.03, 0.04, 0.02, 180, 590)),

    # Fast Food
    "pizza":           ("Fast Food",34, (0.06, 0.29, 0.11, 0.012),   (0.02, 0.04, 0.06, 0.12, 192, 760)),
    "burger":          ("Fast Food",30, (0.07, 0.26, 0.14, 0.014),   (0.01, 0.03, 0.06, 0.10, 230, 820)),
    "tacos":           ("Fast Food",45, (0.07, 0.24, 0.09, 0.018),   (0.02, 0.03, 0.06, 0.08, 200, 560)),
    "bhatura":         ("Fast Food",42, (0.05, 0.28, 0.12, 0.010),   (0.0, 0.0, 0.05, 0.02, 110, 410)),

    # Indian Breads — Extended
    "parota":          ("Grain",    53, (0.04, 0.24, 0.09, 0.012),   (0.0, 0.0, 0.05, 0.02, 95, 320)),
    "appam":           ("Grain",    66, (0.03, 0.19, 0.02, 0.007),   (0.0, 0.0, 0.03, 0.02, 70, 140)),
    "uttapam":         ("Grain",    68, (0.04, 0.18, 0.04, 0.015),   (0.02, 0.01, 0.04, 0.03, 120, 280)),
    "pesarattu":       ("Grain",    72, (0.07, 0.20, 0.03, 0.014),   (0.0, 0.02, 0.05, 0.02, 130, 220)),
    "puttu":           ("Grain",    65, (0.04, 0.22, 0.01, 0.007),   (0.0, 0.0, 0.03, 0.02, 60, 160)),

    # South Indian Extended
    "pongal":          ("Meal",     68, (0.04, 0.21, 0.04, 0.012),   (0.0, 0.0, 0.04, 0.02, 140, 290)),
    "rasam":           ("Soup",     58, (0.02, 0.06, 0.01, 0.008),   (0.12, 0.04, 0.04, 0.02, 200, 480)),

    # North Indian Extended
    "dal_makhani":     ("Meal",     68, (0.08, 0.16, 0.07, 0.036),   (0.04, 0.02, 0.12, 0.06, 380, 520)),
    "palak_paneer":    ("Meal",     72, (0.08, 0.08, 0.12, 0.022),   (0.06, 0.48, 0.10, 0.18, 340, 440)),
    "rajma":           ("Meal",     74, (0.09, 0.20, 0.03, 0.046),   (0.04, 0.02, 0.16, 0.04, 420, 390)),
    "aloo_gobi":       ("Meal",     65, (0.04, 0.16, 0.05, 0.028),   (0.06, 0.01, 0.05, 0.03, 280, 360)),
    "khichdi":         ("Meal",     70, (0.07, 0.20, 0.03, 0.030),   (0.02, 0.01, 0.10, 0.04, 310, 340)),

    # Street Food Extended
    "vada_pav":        ("Fast Food",48, (0.05, 0.28, 0.10, 0.016),   (0.01, 0.01, 0.05, 0.03, 140, 560)),
    "pav_bhaji":       ("Meal",     55, (0.05, 0.22, 0.08, 0.028),   (0.08, 0.06, 0.06, 0.04, 260, 680)),
    "bhel_puri":       ("Meal",     58, (0.04, 0.22, 0.06, 0.020),   (0.04, 0.02, 0.05, 0.02, 180, 480)),
    "halwa":           ("Meal",     42, (0.03, 0.35, 0.08, 0.006),   (0.01, 0.0, 0.04, 0.03, 60, 80)),
    "gulab_jamun":     ("Meal",     35, (0.03, 0.40, 0.07, 0.002),   (0.0, 0.0, 0.02, 0.04, 40, 55)),

    # Healthy / Bowl
    "smoothie_bowl":   ("Bowl",     76, (0.04, 0.18, 0.03, 0.035),   (0.20, 0.08, 0.04, 0.06, 320, 80)),
    "avocado_toast":   ("Meal",     72, (0.05, 0.18, 0.10, 0.048),   (0.08, 0.05, 0.04, 0.02, 360, 280)),
    "oats":            ("Grain",    80, (0.06, 0.24, 0.03, 0.040),   (0.0, 0.0, 0.06, 0.02, 160, 55)),

    # Asian Extended
    "momo":            ("Meal",     60, (0.08, 0.22, 0.06, 0.012),   (0.02, 0.01, 0.06, 0.03, 180, 520)),
    "kebab":           ("Protein",  64, (0.14, 0.04, 0.10, 0.006),   (0.02, 0.01, 0.08, 0.02, 280, 480)),
    "spring_roll":     ("Meal",     53, (0.05, 0.22, 0.10, 0.014),   (0.04, 0.03, 0.04, 0.02, 160, 420)),
}

# Fallback for unknown food classes
_DEFAULT_ENTRY = ("Meal", 55, (0.06, 0.15, 0.05, 0.015), (0.05, 0.05, 0.05, 0.05, 250, 400))

# ---------------------------------------------------------------------------
# ANTIGRAVITY INSIGHT TEMPLATES  (keyed by category)
# Each template is a 4-tuple: (lead_fact, mechanism, optimisation, long_term)
# ---------------------------------------------------------------------------
INSIGHT_TEMPLATES = {
    "Fruit": (
        "Natural fructose in {name} is metabolised primarily in the liver, producing uric acid as a byproduct — moderation preserves hepatic insulin sensitivity.",
        "The polyphenolic pigments (anthocyanins, flavonoids) trigger AMPK activation, mimicking a mild caloric restriction signal at the cellular level.",
        "Pair with a small amount of fat (e.g., almond butter) to enhance carotenoid absorption — fat-soluble phytonutrients require lipid co-transport.",
        "Daily fruit intake correlates with a 28% reduction in cardiovascular mortality risk (NEJM, 2016) — fibre, not sugar, drives this outcome.",
    ),
    "Vegetable": (
        "The glucosinolates in cruciferous vegetables hydrolyse to sulforaphane upon cooking — this activates the Nrf2 antioxidant response element, your cells' most powerful detox switch.",
        "Phytochemicals drive epigenetic suppression of pro-inflammatory NF-κB signalling — one serving of leafy greens can measurably lower IL-6 within 4 hours.",
        "Steam rather than boil to preserve water-soluble vitamins (C, B-group) — boiling leaches up to 55% of Vitamin C into the cooking water.",
        "Habitual vegetable consumption restructures the gut microbiome toward Akkermansia muciniphila-dominant populations, improving mucosal barrier integrity and metabolic flexibility over months.",
    ),
    "Protein": (
        "{name} delivers high biological-value protein, maximising muscle protein synthesis (MPS) — leucine content is the key mTOR trigger, requiring ≥2.5g per meal to surpass the anabolic threshold.",
        "High dietary protein elevates thermic effect of food (TEF) by 20–35% of calories consumed — protein is energetically expensive to digest, passively supporting weight management.",
        "Pair with vitamin C-rich food (lemon squeeze, bell pepper) — ascorbic acid enhances iron bioavailability from animal protein via ferric-to-ferrous reduction.",
        "Regular high-protein meals preserve lean muscle mass during caloric restriction, preventing the metabolic slowdown that causes weight-loss plateau.",
    ),
    "Grain": (
        "The glycaemic load of {name} is moderated by its fibre content — intact grain matrix slows amylase activity, extending glucose release over 60–90 minutes versus refined equivalents.",
        "B-vitamins (thiamine, B6, folate) in whole grains are essential cofactors for ATP synthesis and red blood cell formation — chronic deficiency manifests as fatigue and cognitive fog.",
        "Combine with legumes (dal, lentils) to create a complete amino acid profile — grains are lysine-deficient, legumes are methionine-deficient; together they complement to full biological value.",
        "Replacing refined grains with whole grains for 12 weeks reduces fasting glucose by an average of 6.3 mg/dL in borderline diabetic populations (ADA meta-analysis).",
    ),
    "Salad": (
        "The raw vegetable matrix preserves heat-labile micronutrients — Vitamin C, folate, and enzyme cofactors are delivered intact at levels that cooked forms cannot match.",
        "Mixed-leaf salads generate short-chain fatty acids (SCFAs) via microbial fermentation in the colon — butyrate in particular activates colonic epithelial cells and suppresses colorectal cancer markers.",
        "Add an olive-oil dressing over low-fat variants — fat-soluble carotenoids (lycopene, beta-carotene, lutein) require lipid for micellar transport. Science: fat-free dressing = near-zero carotenoid absorption.",
        "Regular salad consumption is the single strongest dietary predictor of adequate Vitamin K and folate levels, both critical for coagulation cascade regulation and homocysteine metabolism.",
    ),
    "Soup": (
        "Soup's high water content (75–90%) stretches gastric volume, activating stretch receptors and releasing PYY/CCK satiety hormones — functional satiety at lower caloric cost.",
        "Slow-cooked broths release collagen-derived gelatin and glycine — directly supporting intestinal barrier integrity and joint connective tissue repair.",
        "Watch the sodium load in processed soups — exceeding 800mg per serving stimulates aldosterone-driven sodium retention, acutely raising blood pressure by 5–10 mmHg.",
        "Soup-based meal starters reduce total meal caloric intake by an average of 20% in RCT evidence — a practical tool for portion-managed weight regulation strategies.",
    ),
    "Fast Food": (
        "The trans-fat and saturated fat ratio in {name} activates hepatic lipogenesis pathways — chronic exposure elevates LDL-C and apolipoprotein-B, accelerating atherosclerotic plaque formation.",
        "Hyper-palatable formulation (salt+fat+sugar combination) triggers dopaminergic reward circuits disproportionate to nutritional value — repeated intake recalibrates hedonic set-point upward.",
        "If consumed, offset the sodium load ({sodium}mg) with potassium-rich foods at the next meal (banana, avocado, sweet potato) to restore Na⁺/K⁺ pump equilibrium.",
        "Ultra-processed food consumption >4 times/week is independently associated with 62% higher all-cause mortality risk in prospective cohort studies — frequency, not just quantity, matters.",
    ),
    "Meal": (
        "{name} is a composite meal delivering a cross-section of macros — calories, protein, and micronutrients depend heavily on preparation method and portion size.",
        "The glycaemic response is shaped by the protein-fat-fibre matrix present — high-fibre components slow carbohydrate absorption, moderating post-prandial insulin spike.",
        "Assess sodium carefully in restaurant or home-cooked composite meals — seasoning and sauces often contribute 50–70% of the meal's total sodium, not the primary ingredients.",
        "Balanced composite meals that hit the 4:1 carb-to-protein ratio are ideal for post-workout recovery — replenishing glycogen while delivering leucine for acute MPS stimulation.",
    ),
    "Bowl": (
        "Bowl meals integrate multiple macronutrient groups — the macro balance determines metabolic response, with protein and fat content acting as glycaemic brakes on the carbohydrate portion.",
        "Legume and grain combinations typical of bowl-format meals create synergistic amino acid complementarity, approaching the biological value of animal protein.",
        "Increase vegetable variety: each additional plant species per meal contributes new prebiotic fibre strands, diversifying microbiome substrate and expanding metabolic enzyme production.",
        "Frequent consumption of diverse, plant-forward bowl meals is the dietary pattern most consistently associated with longevity in the Blue Zone populations.",
    ),
}

_DEFAULT_INSIGHTS = (
    "The macronutrient structure of {name} yields {calories} kilocalories — analyse the fibre-to-carb ratio to gauge glycaemic impact.",
    "Protein content ({protein}g) supports cellular repair workflows; evaluate co-factor availability (B-vitamins, zinc) to maximise synthesis efficiency.",
    "Micronutrient screen shows {sodium}mg sodium — balance with potassium-rich foods at the next meal to maintain electrolyte homeostasis.",
    "Monitor meal frequency and portion size — caloric density at this level requires careful integration into your daily energy balance target.",
)


def _lookup(food_name: str):
    """Return the KB entry for a food, with fuzzy key matching."""
    name_lower = food_name.lower().replace(" ", "_")
    # Exact match
    if name_lower in FOOD_KB:
        return name_lower, FOOD_KB[name_lower]
    # Partial match
    for key, val in FOOD_KB.items():
        if key in name_lower or name_lower in key:
            return key, val
    return name_lower, _DEFAULT_ENTRY


def _derive_macros(food_name: str, raw_nutrition: dict):
    """Derive full macro profile from raw nutrition data or KB ratios."""
    matched_key, (category, base_score, macro_ratios, _) = _lookup(food_name)
    p_r, c_r, f_r, fi_r = macro_ratios

    cals = raw_nutrition.get('calories', 250)
    protein = raw_nutrition.get('protein', round(cals * p_r, 1))
    carbs   = raw_nutrition.get('carbs',   round(cals * c_r, 1))
    fat     = raw_nutrition.get('fat',     round(cals * f_r, 1))
    fibre   = raw_nutrition.get('fibre',   round(cals * fi_r, 1))

    return {
        "calories":  int(cals),
        "protein_g": round(float(protein), 1),
        "carbs_g":   round(float(carbs), 1),
        "fat_g":     round(float(fat), 1),
        "fibre_g":   round(float(fibre), 1),
    }


def _derive_micros(food_name: str, calories: int):
    """Derive rich micro profile from KB per-food micro factors."""
    _, (_, _, _, micro_profile) = _lookup(food_name)
    vc_f, va_f, fe_f, ca_f, k_base, na_base = micro_profile

    return {
        "vitC_pct":    min(200, int(calories * vc_f + 0.5)),
        "vitA_pct":    min(150, int(calories * va_f + 0.5)),
        "iron_pct":    min(100, int(calories * fe_f + 0.5)),
        "calcium_pct": min(100, int(calories * ca_f + 0.5)),
        "potassium_mg": min(3500, int(k_base + calories * 0.5)),
        "sodium_mg":    min(3000, int(na_base + calories * 0.2)),
    }


def _compute_score(category: str, base_score: int, macros: dict, micros: dict, raw_nutrition: dict) -> int:
    """Compute ANTIGRAVITY 0–100 health score per protocol rubric."""
    score = base_score

    # Bonuses
    if macros["fibre_g"] >= 5:           score += 10
    if macros["protein_g"] >= 20:        score += 10
    if micros["vitC_pct"] >= 100:        score += 5

    # Penalties
    if micros["sodium_mg"] > 800:        score -= 15
    sat_fat_estimate = macros["fat_g"] * 0.35  # rough saturated fat proxy
    if sat_fat_estimate > 15:            score -= 10
    if raw_nutrition.get("sugar", 0) > 20: score -= 15
    if macros["calories"] > 500:         score -= 10

    return max(1, min(100, score))


def _score_label(score: int) -> str:
    if score >= 88:
        return "🟢 Excellent — Top-tier nutritional density. Model food for any diet pattern."
    elif score >= 72:
        return "🟡 Good — Solid nutritional profile."
    elif score >= 50:
        return "🟠 Moderate — Acceptable. Pair with high-fibre vegetables to raise the meal score."
    else:
        return "🔴 Indulge Mindfully — High caloric density and/or processed ingredients. Limit frequency."


def _build_insights(food_name: str, category: str, macros: dict, micros: dict) -> tuple:
    """Return (insights_list, optimisation_tip) using ANTIGRAVITY templates."""
    templates = INSIGHT_TEMPLATES.get(category, INSIGHT_TEMPLATES["Meal"])

    ctx = {
        "name":    food_name,
        "calories": macros["calories"],
        "protein": macros["protein_g"],
        "carbs":   macros["carbs_g"],
        "fat":     macros["fat_g"],
        "fibre":   macros["fibre_g"],
        "sodium":  micros["sodium_mg"],
        "vitC":    micros["vitC_pct"],
        "potassium": micros["potassium_mg"],
    }

    insights = [t.format(**ctx) for t in templates[:4]]
    opt_tip  = (
        f"Your meal delivers {macros['calories']} kcal with {micros['sodium_mg']}mg sodium. "
        f"Pair with {_suggest_pairing(category)} to optimise absorption and balance the nutrient load."
    )
    return insights, opt_tip


def _suggest_pairing(category: str) -> str:
    pairings = {
        "Fruit":     "a handful of walnuts for omega-3 co-transport of fat-soluble vitamins",
        "Vegetable": "a drizzle of extra-virgin olive oil to unlock fat-soluble carotenoids",
        "Protein":   "a squeeze of lemon and leafy greens to enhance non-haem iron absorption via Vitamin C",
        "Grain":     "a serving of dal or legumes to complete the amino acid profile",
        "Salad":     "an olive-oil-based dressing to maximise carotenoid bioavailability",
        "Soup":      "a slice of whole-grain bread for a slow-digesting carbohydrate source",
        "Fast Food": "500ml of water and a high-potassium snack (banana, avocado) to offset the sodium load",
        "Meal":      "a side of steamed cruciferous vegetables to activate Nrf2 detox pathways",
    }
    return pairings.get(category, "a side of leafy greens for micronutrient balance")


def evaluate_meal(nutrition_results: dict, yolo_confidence: float = None) -> dict:
    """
    Main ANTIGRAVITY evaluation engine.
    nutrition_results: dict of {food_name: {calories, protein, carbs, fat, fibre, ...}}
    yolo_confidence: float in [0,1] from YOLO detection, or None to use KB default.
    """
    if not nutrition_results:
        return {"error": "No food detected."}

    main_item_name = list(nutrition_results.keys())[0]
    main_item_data = nutrition_results[main_item_name]

    matched_key, (category, base_score, _, _) = _lookup(main_item_name)

    # Macros & Micros
    macros = _derive_macros(main_item_name, main_item_data)
    micros = _derive_micros(main_item_name, macros["calories"])

    # Health Score
    score = _compute_score(category, base_score, macros, micros, main_item_data)
    score_label = _score_label(score)

    # Insights
    try:
        if generator:
            prompt = (
                f"You are a clinical nutrition AI. Analyze {main_item_name} "
                f"({macros['calories']} kcal, {macros['protein_g']}g protein, "
                f"{micros['sodium_mg']}mg sodium). Write 3 precise scientific insights. "
                "Each starts with a hyphen. No filler phrases. Focus on biochemical mechanisms."
            )
            out = generator(prompt, max_length=200, num_return_sequences=1, truncation=True)
            generated_text = out[0]['generated_text'].replace(prompt, '').strip()
            lines = [l.strip().lstrip('-').strip() for l in generated_text.split('\n') if len(l.strip()) > 15]
            if len(lines) >= 3:
                insights = lines[:4]
                opt_tip = _build_insights(main_item_name, category, macros, micros)[1]
            else:
                raise ValueError("insufficient output")
        else:
            raise ValueError("no generator")
    except Exception:
        insights, opt_tip = _build_insights(main_item_name, category, macros, micros)

    # Confidence
    if yolo_confidence is not None:
        confidence_str = f"{yolo_confidence * 100:.1f}%"
    else:
        confidence_str = f"{main_item_data.get('confidence', 0.95) * 100:.1f}%"

    return {
        "food_detected":   main_item_name.replace("_", " ").title(),
        "confidence":      confidence_str,
        "category":        category,
        "health_score":    score,
        "score_label":     score_label,
        "macros":          macros,
        "micros":          micros,
        "insights":        insights,
        "optimisation_tip": opt_tip,
    }
