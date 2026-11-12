"""
Recycling recommendations per waste category.
"""
from __future__ import annotations

RECOMMENDATIONS: dict[str, str] = {
    "cardboard": (
        "Flatten and bundle cardboard before placing in the recycling bin. "
        "Remove any tape, staples, or plastic inserts. Keep it dry — wet cardboard "
        "is usually not recyclable. Pizza boxes with heavy grease should go in compost."
    ),
    "glass": (
        "Rinse glass bottles and jars before recycling. Remove lids (metal lids "
        "can often be recycled separately). Do NOT include broken glass, mirrors, "
        "window glass, or Pyrex — these have different compositions and contaminate "
        "the glass recycling stream."
    ),
    "metal": (
        "Rinse metal cans to remove food residue. Aluminium and steel cans are "
        "highly recyclable. Crush cans lightly to save space. Aerosol cans are "
        "recyclable when empty. Keep lids inside cans to avoid them getting lost "
        "in sorting machinery."
    ),
    "paper": (
        "Paper is one of the most recyclable materials. Place clean, dry paper in "
        "the recycling bin. Shredded paper may need to be bagged. Avoid recycling "
        "paper contaminated with food, wax coating, or carbon paper."
    ),
    "plastic": (
        "Check the resin identification code (number inside the triangle) on the "
        "plastic item. Types 1 (PET) and 2 (HDPE) are widely recyclable. Types 3–7 "
        "vary by municipality. Rinse containers before recycling. Plastic bags and "
        "film should NOT go in the bin — return them to store drop-off points."
    ),
    "organic": (
        "Collect food scraps, fruit peels, vegetable waste, and coffee grounds in a compost bin. "
        "Do NOT include plastic stickers, wrappers, or meat packaging. Place in the Green Bin "
        "(Wet Waste) for municipal composting or add to your home compost tumbler."
    ),
    "trash": (
        "This item does not appear to fit standard recycling categories. Dispose of "
        "it in your general waste bin. Consider whether it could be repaired, "
        "repurposed, or donated before discarding. Some electronics, batteries, and "
        "hazardous materials require special disposal — check your local guidelines."
    ),
}


def get_recommendation(category: str) -> str:
    """Return a recycling recommendation for the given waste category."""
    return RECOMMENDATIONS.get(
        category.lower(),
        "No specific recommendation available. Please check your local recycling guidelines.",
    )
