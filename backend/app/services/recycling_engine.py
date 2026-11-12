"""
Intelligent Recycling Decision Engine for EcoSort AI V2.

Decouples material classification from waste management and disposal policy.
Provides structured recycling recommendations, preparation steps, safety warnings,
and regional disposal guidance (including India's Swachh Bharat 3-bin system:
Green for Wet/Compostable, Blue for Dry/Recyclable, Black/Red for Domestic Hazardous/Residual).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class DisposalStatus(str, Enum):
    RECYCLABLE = "RECYCLABLE"
    CONDITIONALLY_RECYCLABLE = "CONDITIONALLY_RECYCLABLE"
    COMPOSTABLE = "COMPOSTABLE"
    NOT_STANDARD_RECYCLABLE = "NOT_STANDARD_RECYCLABLE"
    HAZARDOUS_SPECIAL_DISPOSAL = "HAZARDOUS_SPECIAL_DISPOSAL"
    UNKNOWN_REQUIRES_VERIFICATION = "UNKNOWN_REQUIRES_VERIFICATION"


@dataclass
class RecyclingDecision:
    disposal_status: DisposalStatus
    status_label: str
    action: str
    preparation_steps: List[str]
    disposal_bin: str
    reason: str
    safety_info: Optional[str] = None
    regional_caveats: str = "Recycling acceptance rules and bin color codes may vary by municipality."

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["disposal_status"] = self.disposal_status.value
        return data


# Comprehensive material rulebook
MATERIAL_RULES: Dict[str, Dict[str, Any]] = {
    "cardboard": {
        "disposal_status": DisposalStatus.CONDITIONALLY_RECYCLABLE,
        "status_label": "Conditionally Recyclable",
        "action": "Flatten cardboard boxes and remove non-paper packing materials.",
        "preparation_steps": [
            "Remove all plastic tape, packing foam, and shipping labels where feasible.",
            "Flatten boxes completely to save bin volume and transport capacity.",
            "Keep dry — wet or moldy cardboard degrades pulp fibers and cannot be recycled.",
            "Heavily food-greased cardboard (e.g. greasy pizza box bottoms) should be composted or placed in wet waste."
        ],
        "disposal_bin": "Blue Bin (Dry Waste / Paper Recycling)",
        "reason": "Clean corrugated cardboard is highly recyclable and can be pulped into new corrugated medium up to 5-7 times.",
        "safety_info": None,
        "regional_caveats": "Under Indian Municipal Solid Waste (SWM) Rules 2016, clean cardboard is segregated into Dry Waste for authorized kabadiwalas or recyclers.",
    },
    "glass": {
        "disposal_status": DisposalStatus.CONDITIONALLY_RECYCLABLE,
        "status_label": "Conditionally Recyclable",
        "action": "Rinse clean and separate metal/plastic caps before recycling.",
        "preparation_steps": [
            "Empty all liquids and lightly rinse residue.",
            "Remove metal crown caps or plastic screw lids (recycle separately).",
            "Do NOT break bottles before disposal.",
            "Do NOT mix container glass with Pyrex, window glass, mirrors, or incandescent bulbs."
        ],
        "disposal_bin": "Blue Bin (Dry Waste / Glass Collection)",
        "reason": "Container glass is 100% infinitely recyclable without loss of purity or quality.",
        "safety_info": "Broken glass presents a severe laceration hazard for waste sanitation workers. If broken, wrap in thick paper or cardboard and label clearly.",
        "regional_caveats": "Many Indian cities collect whole glass bottles via scrap dealers (kabadiwalas) for deposit refill or cullet smelting.",
    },
    "metal": {
        "disposal_status": DisposalStatus.RECYCLABLE,
        "status_label": "Recyclable",
        "action": "Empty and rinse metal cans and containers.",
        "preparation_steps": [
            "Rinse food or beverage cans to eliminate organic odor and pests.",
            "Lightly compress aluminum beverage cans to conserve space.",
            "Push metal lids down inside food cans or recycle attached.",
            "Aerosol cans must be completely depressurized and empty before recycling."
        ],
        "disposal_bin": "Blue Bin (Dry Waste / Metal Stream)",
        "reason": "Aluminum and steel are closed-loop recyclable materials. Recycling aluminum saves 95% of the energy needed for primary smelting.",
        "safety_info": "Canned food lid edges can be sharp; handle carefully when rinsing.",
        "regional_caveats": "Aluminum cans and tinplate have high scrap commercial value in India; scrap aggregators actively purchase scrap metal.",
    },
    "paper": {
        "disposal_status": DisposalStatus.CONDITIONALLY_RECYCLABLE,
        "status_label": "Conditionally Recyclable",
        "action": "Keep dry and separate from food or liquid waste.",
        "preparation_steps": [
            "Ensure sheets and newspapers are dry and clean.",
            "Remove plastic spiral bindings or metal clips from notebooks.",
            "Avoid sending thermal receipt paper (BPA coated) or wax-coated cartons with standard paper.",
            "Shredded paper should be enclosed in a paper envelope to avoid blowing away."
        ],
        "disposal_bin": "Blue Bin (Dry Waste / Paper)",
        "reason": "Office paper and newsprint are readily converted into recycled newsprint, carton board, and tissue paper.",
        "safety_info": None,
        "regional_caveats": "Wet paper decomposes into pulp slurry that clogs sorting facilities; store dry prior to weekly municipal collection.",
    },
    "plastic": {
        "disposal_status": DisposalStatus.CONDITIONALLY_RECYCLABLE,
        "status_label": "Conditionally Recyclable",
        "action": "Empty, rinse, and verify the Resin Identification Code (RIC #1-#7).",
        "preparation_steps": [
            "Empty contents completely and rinse off food grease or dairy residue.",
            "Inspect the triangular RIC code: #1 PET (water/soda bottles) and #2 HDPE (milk/detergent jugs) are widely recycled.",
            "Codes #3 PVC, #6 Polystyrene (thermocol), and multi-layer plastic pouches are difficult to recycle mechanically.",
            "Crush bottles and screw cap back on if permitted locally."
        ],
        "disposal_bin": "Blue Bin (Dry Waste / Plastics)",
        "reason": "Rigid thermoplastic polymers can be shredded, washed, and pelletized into recycled flakes for polyester yarn or industrial parts.",
        "safety_info": "Never burn plastics — open burning generates toxic dioxins and furans.",
        "regional_caveats": "India's Plastic Waste Management (PWM) Rules mandate Extended Producer Responsibility (EPR) and ban thin single-use plastic films.",
    },
    "organic": {
        "disposal_status": DisposalStatus.COMPOSTABLE,
        "status_label": "Compostable",
        "action": "Place in composting bin or municipal wet waste stream.",
        "preparation_steps": [
            "Collect vegetable peels, fruit rinds, tea leaves, and food scraps in a compostable caddy.",
            "Do NOT mix plastic stickers from fruit, rubber bands, or plastic bags into organic waste.",
            "Drain excess liquid before disposal to prevent anaerobic leachate in bins.",
            "Can be added to backyard aerated compost tumblers or community vermicomposting units."
        ],
        "disposal_bin": "Green Bin (Wet Waste / Biodegradable)",
        "reason": "Biodegradable organic matter naturally decomposes through aerobic composting into nutrient-rich soil humus, cutting methane emissions.",
        "safety_info": "Wash hands after handling spoiled food or decaying organic matter.",
        "regional_caveats": "Mandatory daily segregation into Green Wet Waste bins across Indian municipal corporations (SWM Rules 2016).",
    },
    "trash": {
        "disposal_status": DisposalStatus.NOT_STANDARD_RECYCLABLE,
        "status_label": "Not Standard Recyclable",
        "action": "Dispose in municipal residual / sanitary waste bin.",
        "preparation_steps": [
            "Confirm that this item cannot be repaired, donated, or sorted into recyclable fractions.",
            "If composite item (e.g. foil-plastic laminate, contaminated packaging), place in general non-recyclable refuse.",
            "Seal dusty or sanitary items securely in a bag.",
            "Do NOT dump in open street drains or storm channels."
        ],
        "disposal_bin": "Black Bin (Residual / Non-Recyclable Waste)",
        "reason": "Composite materials, contaminated items, and mixed polymers cannot be economically separated by mechanical recycling lines.",
        "safety_info": "If the item is medical waste, diapers, or sanitary hygiene products, wrap separately in paper bags marked with a red cross.",
        "regional_caveats": "Sent to municipal sanitary landfills or waste-to-energy (WTE) incineration plants.",
    },
    "hazardous": {
        "disposal_status": DisposalStatus.HAZARDOUS_SPECIAL_DISPOSAL,
        "status_label": "Hazardous / Special Disposal",
        "action": "Take to an authorized e-waste or hazardous collection center.",
        "preparation_steps": [
            "Never throw lithium-ion batteries, fluorescent tubes, paint, or chemical solvents in domestic bins.",
            "Tape battery terminals with electrical tape to prevent short circuits and thermal runaway fires.",
            "Keep mercury CFLs unbroken in cushioned packaging.",
            "Deposit at designated e-waste drop-off kiosks, battery retail collection points, or toxic waste drives."
        ],
        "disposal_bin": "Red / Special Hazard Bin (E-Waste / Domestic Hazardous)",
        "reason": "Hazardous heavy metals (lead, cadmium, mercury, lithium) leach into groundwater and pose fire risks to waste trucks.",
        "safety_info": "Corrosive and fire hazard. Handle with gloves and keep away from children.",
        "regional_caveats": "Regulated under India's E-Waste Management Rules 2022 and Hazardous Waste Rules.",
    },
    "unknown": {
        "disposal_status": DisposalStatus.UNKNOWN_REQUIRES_VERIFICATION,
        "status_label": "Needs Verification",
        "action": "Manual inspection required before disposal.",
        "preparation_steps": [
            "The AI classifier detected high uncertainty, ambiguous materials, or an out-of-distribution item.",
            "Examine the item for recycling symbols (RIC codes 1-7, PAP 20, ALU 41, GL 70).",
            "Separate composite components (e.g. glass jar from steel lid, paper box from plastic blister pack).",
            "Consult local municipal waste management portal if unsure."
        ],
        "disposal_bin": "Inspect item before choosing Blue (Dry), Green (Wet), or Black (Residual)",
        "reason": "Ambiguous or multi-material items risk contaminating clean recycling streams if misplaced.",
        "safety_info": "Exercise caution if the item appears to be pressurized, sharp, electronic, or chemical.",
        "regional_caveats": "When in doubt, consult local ragpickers, apartment waste coordinators, or municipal guidelines.",
    },
}


def get_recycling_decision(
    category: str,
    confidence: float = 1.0,
    is_ood: bool = False,
    confidence_status: str = "normal",
    item_context: Optional[str] = None,
) -> RecyclingDecision:
    """
    Compute an intelligent recycling decision based on material category and inference uncertainty.

    If out-of-distribution or confidence is very low, routes to UNKNOWN_REQUIRES_VERIFICATION.
    """
    cat_lower = category.lower().strip()

    # Route uncertainty & OOD to UNKNOWN_REQUIRES_VERIFICATION
    if is_ood or confidence_status in ("out_of_distribution", "unknown") or (confidence < 0.35 and cat_lower != "trash"):
        rule = MATERIAL_RULES["unknown"]
        return RecyclingDecision(
            disposal_status=rule["disposal_status"],
            status_label=rule["status_label"],
            action=rule["action"],
            preparation_steps=list(rule["preparation_steps"]),
            disposal_bin=rule["disposal_bin"],
            reason=f"Classification confidence is low ({confidence*100:.1f}%) or out-of-distribution. {rule['reason']}",
            safety_info=rule["safety_info"],
            regional_caveats=rule["regional_caveats"],
        )

    # Specific material lookup
    rule = MATERIAL_RULES.get(cat_lower, MATERIAL_RULES["trash"])

    # If context indicates hazardous item (battery, chemical, e-waste)
    if item_context and any(kw in item_context.lower() for kw in ["battery", "chemical", "mercury", "bulb", "tube", "electronics"]):
        rule = MATERIAL_RULES["hazardous"]

    return RecyclingDecision(
        disposal_status=rule["disposal_status"],
        status_label=rule["status_label"],
        action=rule["action"],
        preparation_steps=list(rule["preparation_steps"]),
        disposal_bin=rule["disposal_bin"],
        reason=rule["reason"],
        safety_info=rule.get("safety_info"),
        regional_caveats=rule["regional_caveats"],
    )
