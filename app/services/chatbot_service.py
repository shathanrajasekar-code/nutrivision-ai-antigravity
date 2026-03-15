from transformers import pipeline

try:
    chatbot = pipeline('text-generation', model='HuggingFaceTB/SmolLM-135M', max_new_tokens=120)
except Exception:
    chatbot = None

# ---------------------------------------------------------------------------
# INTENT ROUTING — keyword → depth-level guidance injected into prompt
# ---------------------------------------------------------------------------
INTENT_GUIDANCE = {
    "protein": (
        "PROTEIN QUERY. Respond with: (1) specific high-protein foods with exact gram amounts per serving, "
        "(2) the leucine threshold for mTOR activation (≥2.5g/meal), "
        "(3) how protein timing post-workout (within 2h) maximises muscle protein synthesis."
    ),
    "weight": (
        "WEIGHT MANAGEMENT QUERY. Respond with: (1) caloric deficit mechanics (500 kcal/day = ~0.5kg/week), "
        "(2) the metabolic adaptation risk of aggressive restriction (<1200 kcal), "
        "(3) foods that maximise satiety per calorie (fibre + protein density)."
    ),
    "energy": (
        "ENERGY QUERY. Respond with: (1) the role of B-vitamins (B1, B2, B3, B5) in the Krebs cycle, "
        "(2) iron-deficiency anaemia as the most common energy-drain mechanism, "
        "(3) blood glucose stability strategies (low-GI meals, avoiding sugar spikes)."
    ),
    "inflammation": (
        "INFLAMMATION QUERY. Respond with: (1) the omega-6/omega-3 ratio and its control of eicosanoid balance, "
        "(2) specific anti-inflammatory foods (fatty fish, turmeric/curcumin, walnuts, blueberries), "
        "(3) how ultra-processed foods activate NF-κB inflammatory pathways."
    ),
    "gut": (
        "GUT HEALTH QUERY. Respond with: (1) the role of prebiotic fibre in feeding Lactobacillus and Bifidobacterium, "
        "(2) the SCFA butyrate as the primary fuel for colonocytes and its anti-cancer properties, "
        "(3) foods that harm the microbiome (emulsifiers, artificial sweeteners, low-fibre diets)."
    ),
    "sugar": (
        "BLOOD SUGAR / GLYCAEMIC QUERY. Respond with: (1) how the glycaemic index differs from glycaemic load, "
        "(2) the cephalic phase insulin response and why sweet taste alone can spike insulin, "
        "(3) chromium and magnesium's role in insulin receptor sensitivity."
    ),
    "vitamin": (
        "VITAMIN QUERY. Respond with: (1) the specific deficiency marker for the relevant vitamin, "
        "(2) best food sources with bioavailability rates, "
        "(3) interaction effects (e.g., Vitamin D requires fat; iron blocks zinc at high doses)."
    ),
    "sleep": (
        "SLEEP & NUTRITION QUERY. Respond with: (1) tryptophan-to-serotonin-to-melatonin synthesis pathway and food sources, "
        "(2) magnesium glycinate's role in GABA potentiation and deep sleep architecture, "
        "(3) foods/timing to avoid before bed (high glycaemic, caffeine half-life)."
    ),
}

# ---------------------------------------------------------------------------
# FALLBACK RESPONSES — deterministic ANTIGRAVITY-grade answers by intent
# ---------------------------------------------------------------------------
FALLBACK_RESPONSES = {
    "protein": (
        "To increase dietary protein meaningfully:\n\n"
        "• **Eggs** (6g/egg) — complete amino acid profile, high leucine content triggers mTOR activation.\n"
        "• **Greek yoghurt** (17g/200g) — high casein fraction for slow-release overnight muscle repair.\n"
        "• **Chicken breast** (31g/100g) — leanest high-protein source; pair with Vitamin C to enhance iron co-absorption.\n"
        "• **Lentils** (9g/100g cooked) — plant-based; combine with rice for full amino acid complementarity.\n\n"
        "Target ≥2.5g leucine per meal to cross the anabolic threshold. This requires ~30g protein from most sources."
    ),
    "weight": (
        "Weight management rests on energy balance with a nuance: diet quality determines whether you lose fat or muscle.\n\n"
        "• A 500 kcal daily deficit produces ~0.5kg/week fat loss — sustainable and muscle-preserving.\n"
        "• Below 1200 kcal, basal metabolic rate adapts downward via thyroid hormone suppression — the plateau effect.\n"
        "• Fill volume with high-satiety foods: oats (β-glucan expands in stomach), eggs (CCK release), broccoli (low density, high fibre).\n\n"
        "Prioritise protein at 1.6–2.2g/kg body weight to preserve lean mass during restriction."
    ),
    "inflammation": (
        "Chronic low-grade inflammation is driven primarily by dietary lipid ratios.\n\n"
        "• Modern diets carry an omega-6/omega-3 ratio of ~15:1. Target is 4:1 or lower.\n"
        "• Top anti-inflammatory foods: **fatty fish** (EPA/DHA directly suppress prostaglandin E2), **walnuts** (ALA precursor), **turmeric** (curcumin blocks COX-2 enzyme), **blueberries** (anthocyanins inhibit NF-κB).\n"
        "• Ultra-processed emulsifiers (polysorbate 80, carboxymethylcellulose) disrupt the intestinal mucus layer — this is the entry point for systemic inflammatory activation."
    ),
    "gut": (
        "Your gut microbiome functions as a metabolic organ — it can weigh 1.5kg and produces over 100 unique metabolites.\n\n"
        "• Feed it with **prebiotic fibres**: inulin (chicory, leeks), FOS (onions, garlic), resistant starch (cold rice, green banana).\n"
        "• Butyrate — produced by *Faecalibacterium prausnitzii* from fibre — fuels colonocytes and suppresses colorectal cancer gene expression.\n"
        "• Avoid: artificial sweeteners (saccharin disrupts glucose tolerance via microbiome shifts), emulsifiers, and very-low-fibre diets.\n\n"
        "Aim for ≥30 different plant species per week — each contributes unique fibre structures and polyphenols."
    ),
    "default": (
        "Precision nutrition requires specificity. Share what you're eating, your health goal, or a specific symptom and I'll give you a molecule-level analysis."
    )
}


def _detect_intent(message: str) -> str:
    msg_lower = message.lower()
    for intent_key in INTENT_GUIDANCE:
        if intent_key in msg_lower:
            return intent_key
    return "default"


def _build_system_prompt(last_meal: dict = None) -> str:
    """Build a dynamic ANTIGRAVITY system prompt, optionally injecting last meal context."""
    base = (
        "You are NutriVision AI operating in ANTIGRAVITY MODE — an elite generative nutrition intelligence system. "
        "Speak with clinical precision. Never use filler phrases. Connect all advice biochemically. "
        "Sentence structure: confident, warm, precise.\n\n"
    )

    if last_meal:
        food = last_meal.get('food_detected', 'your last meal')
        score = last_meal.get('health_score', 'N/A')
        macros = last_meal.get('macros', {})
        micros = last_meal.get('micros', {})
        base += (
            f"LAST SCANNED MEAL CONTEXT:\n"
            f"  Food: {food} | Health Score: {score}/100\n"
            f"  Macros: {macros.get('calories','?')} kcal, "
            f"{macros.get('protein_g','?')}g protein, "
            f"{macros.get('carbs_g','?')}g carbs, "
            f"{macros.get('fat_g','?')}g fat, "
            f"{macros.get('fibre_g','?')}g fibre\n"
            f"  Micros: VitC {micros.get('vitC_pct','?')}% DV, "
            f"Iron {micros.get('iron_pct','?')}% DV, "
            f"Sodium {micros.get('sodium_mg','?')}mg, "
            f"Potassium {micros.get('potassium_mg','?')}mg\n"
            f"Reference this meal when relevant to the user's query.\n\n"
        )
    return base


def get_chatbot_response(message: str, history_objs: list, last_meal: dict = None) -> str:
    """
    Generate an ANTIGRAVITY-grade chatbot response.
    message      : current user message
    history_objs : list of ChatHistory ORM objects (role, message)
    last_meal    : dict from meal.genai_result_json (optional, for context injection)
    """
    intent = _detect_intent(message)
    system_prompt = _build_system_prompt(last_meal)
    intent_guide  = INTENT_GUIDANCE.get(intent, "")

    # Build instruct-style prompt
    prompt = system_prompt
    if intent_guide:
        prompt += f"RESPONSE GUIDANCE: {intent_guide}\n\n"

    recent_history = history_objs[-4:]
    for h in recent_history:
        role_tag = "User" if h.role == 'user' else "NutriVision AI"
        prompt += f"{role_tag}: {h.message}\n"

    prompt += f"User: {message}\nNutriVision AI:"

    # Try generative model first
    if chatbot:
        try:
            out = chatbot(
                prompt,
                max_length=min(len(prompt.split()) + 150, 400),
                num_return_sequences=1,
                pad_token_id=50256,
                truncation=True
            )
            raw = out[0]['generated_text']
            # Extract only the AI's last response
            if "NutriVision AI:" in raw:
                response = raw.split("NutriVision AI:")[-1].strip()
                # Trim at the next "User:" if present
                if "User:" in response:
                    response = response.split("User:")[0].strip()
                if len(response) > 30:
                    return response
        except Exception:
            pass

    # Fallback to deterministic ANTIGRAVITY response
    fallback = FALLBACK_RESPONSES.get(intent, FALLBACK_RESPONSES["default"])

    # If there's meal context and the model fell back, prepend meal ref
    if last_meal and intent in ("protein", "weight", "energy"):
        food = last_meal.get('food_detected', 'your last meal')
        score = last_meal.get('health_score')
        macros = last_meal.get('macros', {})
        prefix = (
            f"Based on your last scan — **{food}** (Score: {score}/100, "
            f"{macros.get('calories','?')} kcal, {macros.get('protein_g','?')}g protein) — "
            f"here's the precision guidance:\n\n"
        )
        return prefix + fallback

    return fallback
