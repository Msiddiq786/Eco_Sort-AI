import type { RecyclingGuideMaterial } from '@/types'

export const RECYCLING_GUIDE_MATERIALS: RecyclingGuideMaterial[] = [
  // ── 1. CARDBOARD ────────────────────────────────────────────────────────
  {
    id: 'cardboard-corrugated',
    name: 'Corrugated Shipping Boxes',
    category: 'cardboard',
    recyclingPossibility: 'Widely Recyclable',
    badgeVariant: 'success',
    description:
      'Multi-layer corrugated cardboard boxes used for e-commerce deliveries, moving boxes, and shipping containers. High-quality unbleached wood fibers that can be recycled multiple times.',
    commonExamples: [
      'Amazon shipping cartons',
      'Moving & storage boxes',
      'Appliance packaging cartons',
      'Shoe boxes (corrugated)',
    ],
    preparationInstructions: [
      'Remove packing materials such as bubble wrap, foam peanuts, and air pillows.',
      'Peel off excessive plastic shipping tape and shipping label pouches where easily removable.',
      'Flatten all boxes completely to conserve space in recycling bins and collection trucks.',
      'Ensure the cardboard is dry; wet fibers degrade and mold easily.',
    ],
    disposalMethod: {
      binName: 'Cardboard / Mixed Recycling Bin',
      binColor: '#D4A54A',
      instructions: 'Place flattened dry cardboard inside the recycling cart or bundle neatly beside it per municipal rules.',
    },
    avoidMixing: [
      'Do not mix with food-stained or grease-soaked cardboard (e.g. cheesy pizza box bases).',
      'Do not include wax-coated produce boxes or boxes contaminated with paint or chemicals.',
    ],
    environmentalBenefits:
      'Recycling corrugated cardboard uses 75% less energy than manufacturing virgin cardboard from wood pulp and significantly reduces commercial landfill mass.',
    safetyWarnings:
      'Inspect boxes for stray packing staples or box-cutter blades left inside from unpacking.',
    iconName: 'Package',
  },
  {
    id: 'cardboard-paperboard',
    name: 'Paperboard & Cereal Boxes',
    category: 'cardboard',
    recyclingPossibility: 'Widely Recyclable',
    badgeVariant: 'success',
    description:
      'Single-layer paperboard and chipboard used for dry grocery packaging, shoe boxes, tissue boxes, and paper towel core tubes.',
    commonExamples: [
      'Cereal & oatmeal boxes',
      'Pasta, cracker & snack boxes',
      'Tissue boxes (remove plastic film around slot)',
      'Toilet paper & paper towel inner tubes',
    ],
    preparationInstructions: [
      'Remove and discard interior plastic bags or foil liners (e.g. cereal bag liners).',
      'Flatten boxes completely before placing into the recycling bin.',
      'Keep dry and clean from liquid spills.',
    ],
    disposalMethod: {
      binName: 'Paper & Cardboard Bin',
      binColor: '#D4A54A',
      instructions: 'Flatten and drop loose into the paper/cardboard recycling cart.',
    },
    avoidMixing: [
      'Do not include frozen food boxes with plastic moisture barriers unless explicitly marked recyclable locally.',
      'Avoid paperboard with heavily metallic or holographic decorative coatings.',
    ],
    environmentalBenefits:
      'Recycled paperboard is repulped into new cereal boxes, tissue boxes, and building insulation materials.',
    iconName: 'Box',
  },
  {
    id: 'cardboard-pizza-boxes',
    name: 'Pizza Boxes (Clean vs Soiled)',
    category: 'cardboard',
    recyclingPossibility: 'Conditionally Recyclable',
    badgeVariant: 'warning',
    description:
      'Cardboard pizza boxes often have a clean, oil-free top lid and a heavily grease-soaked, cheese-stained bottom.',
    commonExamples: [
      'Takeout delivery pizza boxes',
      'Flatbread packaging boxes',
      'Bakery cardboard pastry boxes',
    ],
    preparationInstructions: [
      'Inspect the box: separate the clean top half from the grease-stained bottom half.',
      'Tear off the clean lid and place it in your cardboard recycling bin.',
      'Discard the oily, food-contaminated bottom in the trash or municipal organics/compost bin if accepted.',
      'Remove all pizza residue, garlic sauce cups, wax liners, and plastic lid supports (pizza savers).',
    ],
    disposalMethod: {
      binName: 'Split: Cardboard (Top) / Organics or Trash (Bottom)',
      binColor: '#D4A54A',
      instructions: 'Recycle the clean cardboard lid; compost or discard the greasy bottom.',
    },
    avoidMixing: [
      'Never put grease-soaked cardboard into paper recycling; oil cannot be separated from paper pulp during water-based repulping and ruins entire batches.',
    ],
    environmentalBenefits:
      'Composting greasy cardboard provides valuable carbon (brown matter) for soil conditioning without contaminating paper recycling streams.',
    iconName: 'Pizza',
  },

  // ── 2. GLASS ────────────────────────────────────────────────────────────
  {
    id: 'glass-container-bottles',
    name: 'Beverage Glass Bottles & Food Jars',
    category: 'glass',
    recyclingPossibility: 'Widely Recyclable',
    badgeVariant: 'success',
    description:
      'Container glass made from soda-lime silica. 100% infinitely recyclable without any loss of purity, strength, or structural quality.',
    commonExamples: [
      'Clear, brown, and green beverage bottles',
      'Jam, jelly, pasta sauce & condiment jars',
      'Olive oil & vinegar glass containers',
      'Pickle jars and salsa jars',
    ],
    preparationInstructions: [
      'Empty all residual liquids and food contents.',
      'Give a quick rinse with minimal water to remove food residue and odors.',
      'Labels and adhesive stickers do not need to be scraped off; they burn off during furnace cullet melting.',
      'Remove metal screw caps (metal caps can be recycled separately with metals).',
    ],
    disposalMethod: {
      binName: 'Glass Recycling Bin / Mixed Curbside',
      binColor: '#0EA5E9',
      instructions: 'Place loose glass bottles into the designated glass or commingled recycling bin. Do not bag in plastic.',
    },
    avoidMixing: [
      'Do not mix with heat-resistant borosilicate glassware (Pyrex), ceramics, window panes, light bulbs, or drinking glasses.',
    ],
    environmentalBenefits:
      'Recycling 1 ton of glass saves 1,300 lbs of sand and 410 lbs of soda ash, reducing industrial furnace carbon emissions by 20%.',
    iconName: 'Wine',
  },
  {
    id: 'glass-non-recyclable',
    name: 'Ceramics, Window Glass & Pyrex',
    category: 'glass',
    recyclingPossibility: 'Residual Waste (Non-Recyclable)',
    badgeVariant: 'neutral',
    description:
      'Non-container glass and ceramics have different chemical compositions and much higher melting points than bottle glass.',
    commonExamples: [
      'Pyrex, Corningware, and tempered ovenware',
      'Window pane glass and automotive windshields',
      'Mirrors and picture frame glass',
      'Ceramic coffee mugs, porcelain plates, and vases',
      'Drinking tumblers and crystal stemware',
    ],
    preparationInstructions: [
      'If in good condition, consider donating to thrift stores or community kitchens.',
      'If broken, wrap securely in several layers of newspaper or cardboard before placing into residual trash to protect waste collectors.',
    ],
    disposalMethod: {
      binName: 'Residual Trash Bin (Safely Wrapped)',
      binColor: '#6B7280',
      instructions: 'Wrap sharp shards securely in paper/tape and discard in ordinary household trash.',
    },
    avoidMixing: [
      'Never put ceramics or window glass into glass recycling bins; even a single ceramic cup can ruin an entire batch of molten glass cullet.',
    ],
    environmentalBenefits:
      'Safe containment protects municipal sanitation workers from dangerous lacerations.',
    safetyWarnings:
      'CAUTION: Broken glass and ceramics present severe cut hazards. Always package sharp shards securely.',
    iconName: 'AlertTriangle',
  },

  // ── 3. METAL ────────────────────────────────────────────────────────────
  {
    id: 'metal-aluminum-cans',
    name: 'Aluminum Beverage Cans',
    category: 'metal',
    recyclingPossibility: 'Widely Recyclable',
    badgeVariant: 'success',
    description:
      'Lightweight, high-value aluminum beverage cans. Aluminum is among the most cost-effective and energy-efficient materials to recycle in the circular economy.',
    commonExamples: [
      'Soda, seltzer, and soft drink cans',
      'Energy drink and iced tea cans',
      'Beer and cider cans',
      'Sparkling water aluminum cans',
    ],
    preparationInstructions: [
      'Empty all liquid contents completely.',
      'A brief rinse is recommended to eliminate sticky syrup residue and prevent insect attraction.',
      'Keep the aluminum pull tab attached.',
      'Check local guidelines before crushing: automated sorting facilities often prefer uncrushed cans for 3D optical recognition.',
    ],
    disposalMethod: {
      binName: 'Metal / Mixed Recycling Bin',
      binColor: '#64748B',
      instructions: 'Place clean, empty cans loose into the recycling container.',
    },
    avoidMixing: [
      'Do not mix with cans containing hazardous liquids, gasoline, paint thinner, or pesticides.',
    ],
    environmentalBenefits:
      'Recycling aluminum saves 95% of the energy needed to smelt virgin aluminum from bauxite ore, returning to store shelves as new cans in 60 days.',
    iconName: 'Hammer',
  },
  {
    id: 'metal-steel-cans',
    name: 'Steel & Tin Food Cans',
    category: 'metal',
    recyclingPossibility: 'Widely Recyclable',
    badgeVariant: 'success',
    description:
      'Steel and tin-plated steel food cans used for preserved vegetables, soups, beans, pet food, and tomato sauce. Highly magnetic and easy for recycling facilities to separate.',
    commonExamples: [
      'Soup and broth cans',
      'Canned beans, corn, and vegetable tins',
      'Canned tuna and pet food tins',
      'Metal bottle caps and metal jar lids',
    ],
    preparationInstructions: [
      'Empty all food residue and give a quick rinse.',
      'Paper labels can remain on; they combust cleanly during high-temperature steel smelting.',
      'If you have detached the metal lid, place it inside the can and crimp the rim slightly to trap it safely inside.',
    ],
    disposalMethod: {
      binName: 'Metal / Commingled Recycling Bin',
      binColor: '#64748B',
      instructions: 'Place into your curbside recycling container.',
    },
    avoidMixing: [
      'Avoid cans containing motor oil, wet paint, or flammable solvents.',
    ],
    environmentalBenefits:
      'Every ton of recycled steel saves 2,500 lbs of iron ore, 1,400 lbs of coal, and 120 lbs of limestone.',
    safetyWarnings:
      'Watch out for sharp metal can rims and lid edges after opening.',
    iconName: 'Shield',
  },
  {
    id: 'metal-aerosols',
    name: 'Aerosol Spray Cans',
    category: 'metal',
    recyclingPossibility: 'Conditionally Recyclable',
    badgeVariant: 'warning',
    description:
      'Steel and aluminum pressurized aerosol spray cans for deodorant, shaving cream, whipped cream, and hairspray.',
    commonExamples: [
      'Deodorant and body spray cans',
      'Hair spray and dry shampoo cans',
      'Cooking oil spray cans',
      'Shaving cream cans',
    ],
    preparationInstructions: [
      'The can MUST be completely empty with zero propellant or pressure remaining.',
      'Press the nozzle until no hissing sound or product is emitted.',
      'Remove the exterior plastic cap and recycle with plastics if accepted.',
      'Never puncture, crush, or heat aerosol cans.',
    ],
    disposalMethod: {
      binName: 'Metal Recycling (Only if 100% Empty) / Hazardous Drop-off',
      binColor: '#64748B',
      instructions: 'Recycle empty cans in metal stream if accepted locally. If partially full, take to a hazardous waste depot.',
    },
    avoidMixing: [
      'Never recycle partially full aerosol cans or cans containing spray paint, insect killer, or industrial adhesives in curbside bins.',
    ],
    environmentalBenefits:
      'Recovers high-purity aluminum and steel alloys while preventing dangerous landfill gas release.',
    safetyWarnings:
      'DANGER: Partially full aerosol cans can explode and cause fires in recycling collection trucks and baling compactors.',
    iconName: 'Flame',
  },

  // ── 4. PAPER ────────────────────────────────────────────────────────────
  {
    id: 'paper-clean-office',
    name: 'Clean Office Paper & Documents',
    category: 'paper',
    recyclingPossibility: 'Widely Recyclable',
    badgeVariant: 'success',
    description:
      'High-grade bleached wood cellulose fibers from office printing, copy paper, notebooks, mail envelopes, and stationery.',
    commonExamples: [
      'Printer and copy paper sheets',
      'Notebook pages and lined paper',
      'Envelopes (plastic address windows are acceptable in modern paper mills)',
      'Direct mail advertisements and paper brochures',
    ],
    preparationInstructions: [
      'Keep dry at all costs; water ruins paper fiber structure and causes instant mildew in paper bales.',
      'Small staples and paper clips do not need to be removed; industrial pulpers screen them out magnetically.',
      'Remove plastic spiral bindings or heavy metal ring binders before recycling.',
    ],
    disposalMethod: {
      binName: 'Paper / Dry Recyclables Bin',
      binColor: '#10B981',
      instructions: 'Place loosely into the paper recycling cart.',
    },
    avoidMixing: [
      'Do not include paper soiled with food, oil, paint, or bodily fluids.',
      'Do not mix with wax paper, parchment paper, or thermal receipts.',
    ],
    environmentalBenefits:
      'Recycling 1 ton of office paper saves 17 mature trees, 7,000 gallons of water, and 4,000 kilowatt-hours of electrical power.',
    iconName: 'FileText',
  },
  {
    id: 'paper-newsprint-magazines',
    name: 'Newspapers, Magazines & Catalogs',
    category: 'paper',
    recyclingPossibility: 'Widely Recyclable',
    badgeVariant: 'success',
    description:
      'Printed newsprint, glossy consumer magazines, sales catalogs, and telephone directories.',
    commonExamples: [
      'Daily and weekly newspapers',
      'Glossy fashion and lifestyle magazines',
      'Store circulars and coupon flyers',
      'Paperback books and catalogs',
    ],
    preparationInstructions: [
      'Remove plastic mailing sleeve wrappers and subscription sample pouches (e.g. perfume foils).',
      'Keep paper dry and bundle loosely.',
      'Cardstock inserts can remain inside.',
    ],
    disposalMethod: {
      binName: 'Paper Recycling Bin',
      binColor: '#10B981',
      instructions: 'Place into paper cart. Avoid tying with plastic string or nylon cords.',
    },
    avoidMixing: [
      'Do not include plastic wrap, foil-laminated covers, or wet newsprint.',
    ],
    environmentalBenefits:
      'De-inked newsprint is remade into fresh newsprint and egg cartons within weeks.',
    iconName: 'BookOpen',
  },
  {
    id: 'paper-thermal-receipts',
    name: 'Thermal Receipts & Soiled Paper',
    category: 'paper',
    recyclingPossibility: 'Residual Waste (Non-Recyclable)',
    badgeVariant: 'neutral',
    description:
      'Cash register receipts are printed on thermal paper coated with chemical color-formers (Bisphenol A or S). Used paper towels and tissues contain wet-strength resins and hygiene contaminants.',
    commonExamples: [
      'Store checkout register receipts',
      'ATM transaction slips',
      'Used paper towels and napkins',
      'Facial tissues and wet wipes',
      'Laminated paper sheets and metallic gift wrap',
    ],
    preparationInstructions: [
      'Opt for digital e-receipts sent by email or SMS where possible.',
      'Discard thermal slips and soiled hygiene paper in regular residual trash.',
    ],
    disposalMethod: {
      binName: 'Residual Trash Bin / Organic Compost (for unbleached napkins)',
      binColor: '#6B7280',
      instructions: 'Discard with household residual trash. Unbleached plain napkins can be composted in organics bins.',
    },
    avoidMixing: [
      'NEVER put thermal receipts into paper recycling; BPA chemicals contaminate recycled paper products like toilet paper and food cartons.',
    ],
    environmentalBenefits:
      'Prevents endocrine-disrupting BPA and BPS chemical contamination from entering recycled paper manufacturing streams.',
    iconName: 'Receipt',
  },

  // ── 5. PLASTIC ──────────────────────────────────────────────────────────
  {
    id: 'plastic-pet-bottles',
    name: 'PET Plastic Bottles (#1 PET)',
    category: 'plastic',
    recyclingPossibility: 'Widely Recyclable',
    badgeVariant: 'success',
    description:
      'Polyethylene Terephthalate (#1), the most commonly recycled clear plastic container in the world. Used for water, soda, sports drinks, and clear condiment bottles.',
    commonExamples: [
      'Clear plastic water and soda bottles',
      'Sports hydration drink bottles',
      'Clear salad dressing and mouthwash bottles',
      'Transparent clamshell berry containers (check local acceptance)',
    ],
    preparationInstructions: [
      'Empty all liquids completely.',
      'Rinse lightly to remove sweet syrups and sticky residue.',
      'Crush the bottle slightly to save bin space.',
      'Screw the plastic cap tightly back onto the bottle; caps are made of HDPE and modern float-sink sorting facilities separate bottle flakes from cap flakes.',
    ],
    disposalMethod: {
      binName: 'Plastic / Commingled Recycling Bin',
      binColor: '#F97316',
      instructions: 'Deposit loose into curbside recycling. Do not bundle in plastic bags.',
    },
    avoidMixing: [
      'Do not include plastic squeeze pouches with foil liners or plastic toys.',
    ],
    environmentalBenefits:
      'Recycled PET (rPET) is remanufactured into new beverage bottles, fleece jackets, polyester carpet, and thermal insulation.',
    iconName: 'Milk',
  },
  {
    id: 'plastic-hdpe-jugs',
    name: 'HDPE Rigid Jugs & Bottles (#2 HDPE)',
    category: 'plastic',
    recyclingPossibility: 'Widely Recyclable',
    badgeVariant: 'success',
    description:
      'High-Density Polyethylene (#2), durable, chemical-resistant opaque plastic used for milk jugs, shampoo bottles, and laundry detergents.',
    commonExamples: [
      'Translucent gallon milk and water jugs',
      'Laundry detergent and fabric softener bottles',
      'Shampoo, conditioner, and body wash bottles',
      'Bleach and household cleaning jugs',
    ],
    preparationInstructions: [
      'Empty liquid contents thoroughly.',
      'Rinse out remaining soap film or milk residue.',
      'Replace the screw cap onto the clean bottle.',
      'Remove pump dispensers containing metal springs (pumps go into trash).',
    ],
    disposalMethod: {
      binName: 'Plastic Recycling Bin',
      binColor: '#F97316',
      instructions: 'Place into curbside plastics recycling bin.',
    },
    avoidMixing: [
      'Do not mix with motor oil jugs; oil penetrates HDPE plastic pores and ruins recycled resin.',
    ],
    environmentalBenefits:
      'Recycled HDPE is molded into heavy-duty plastic lumber, park benches, recycling bins, and drainage pipes.',
    iconName: 'Droplets',
  },
  {
    id: 'plastic-soft-film',
    name: 'Plastic Bags, Film & Wraps (#4 LDPE)',
    category: 'plastic',
    recyclingPossibility: 'Special Disposal Required',
    badgeVariant: 'info',
    description:
      'Low-density polyethylene film used for retail shopping bags, bubble wrap, dry cleaning bags, and bread bags. Highly recyclable ONLY through specialized drop-off points.',
    commonExamples: [
      'Supermarket grocery checkout bags',
      'Bread bags and produce bags',
      'Bubble wrap and plastic air shipping pillows (deflated)',
      'Case wrap around water bottle packs and paper towel rolls',
      'Dry cleaner bags and newspaper plastic sleeves',
    ],
    preparationInstructions: [
      'Ensure bags are completely clean, dry, and free of food crumbs or receipts.',
      'Pop and deflate all air shipping pillows.',
      'Collect multiple bags into one bag for easy handling.',
      'NEVER put loose or bagged plastic film into curbside recycling carts.',
    ],
    disposalMethod: {
      binName: 'Store Drop-off Recycling Station',
      binColor: '#0EA5E9',
      instructions: 'Bring to plastic film drop-off bins located at the entrance of major grocery stores and supermarkets.',
    },
    avoidMixing: [
      'NEVER put in home curbside recycling bins! Soft film tangles around high-speed sorting rollers and shuts down municipal MRF facilities.',
      'Do not include crunchy chip bags, salad bags with moisture coatings, or frozen food bags.',
    ],
    environmentalBenefits:
      'Recycled clean film is manufactured into composite decking (like Trex), outdoor furniture, and new commercial trash liners.',
    safetyWarnings:
      'WARNING: Curbside tangler! Plastic film wrap causes costly machinery jams and dangerous manual unjamming operations at sorting plants.',
    iconName: 'ShoppingBag',
  },

  // ── 6. ORGANIC ──────────────────────────────────────────────────────────
  {
    id: 'organic-food-scraps',
    name: 'Fruit, Vegetable & Food Scraps',
    category: 'organic',
    recyclingPossibility: 'Compostable',
    badgeVariant: 'success',
    description:
      'Nutrient-rich biodegradable food remnants and kitchen prep waste. Diverting food waste to composting creates fertile organic humus and eliminates methane emissions from landfills.',
    commonExamples: [
      'Fruit peels, apple cores, and banana skins',
      'Vegetable cuttings, carrot peels, and potato skins',
      'Watermelon rinds and citrus peels',
      'Crushed eggshells',
      'Coffee grounds, paper coffee filters, and loose tea leaves',
      'Cooked leftovers, stale bread, and pasta (check local compost rules)',
    ],
    preparationInstructions: [
      'Remove all non-biodegradable items: PLASTIC PRODUCE STICKERS, rubber bands, and twist ties.',
      'Do not bag food scraps in ordinary plastic bags; use certified compostable liners or paper bags if required.',
      'Keep meat, bones, and dairy separate unless your municipality uses high-temperature industrial composting.',
    ],
    disposalMethod: {
      binName: 'Green Bin / Wet Waste / Organics Cart / Home Compost',
      binColor: '#84CC16',
      instructions: 'Deposit into your municipal green organics cart, kitchen countertop caddy, or backyard compost pile.',
    },
    avoidMixing: [
      'NEVER mix with plastic bags, plastic produce stickers, foil wrappers, diapers, or pet feces.',
      'Do not confuse ordinary "degradable" plastics with certified compostable materials.',
    ],
    environmentalBenefits:
      'Composting food waste prevents anaerobic decomposition in landfills, reducing potent methane emissions (CH4 has 28x the warming potential of CO2) and generating nutrient-dense organic fertilizer.',
    iconName: 'Apple',
  },
  {
    id: 'organic-yard-waste',
    name: 'Garden & Yard Trimmings',
    category: 'organic',
    recyclingPossibility: 'Compostable',
    badgeVariant: 'success',
    description:
      'Natural organic biomass from landscaping, garden maintenance, and lawn mowing.',
    commonExamples: [
      'Fallen leaves and garden clippings',
      'Lawn grass clippings',
      'Pruned shrub branches and twigs (under 2 inches diameter)',
      'Spent flowers, weeds, and garden potting soil',
    ],
    preparationInstructions: [
      'Ensure branches are cut to lengths conforming to local collection limits (typically under 4 feet).',
      'Do not include treated lumber, fencing, rocks, plastic plant pots, or synthetic soil bags.',
    ],
    disposalMethod: {
      binName: 'Yard Waste / Brown Paper Bags / Green Bin',
      binColor: '#84CC16',
      instructions: 'Place into municipal yard waste carts or bundle with biodegradable natural twine.',
    },
    avoidMixing: [
      'Do not use synthetic plastic bags or nylon zip ties.',
      'Avoid invasive weeds with mature seeds or plants treated with persistent herbicides.',
    ],
    environmentalBenefits:
      'Converted into rich municipal mulch and soil conditioners distributed to city parks and local farms.',
    iconName: 'Leaf',
  },

  // ── 7. TRASH ────────────────────────────────────────────────────────────
  {
    id: 'trash-multi-layer',
    name: 'Multi-layer Wrappers & Snack Bags',
    category: 'trash',
    recyclingPossibility: 'Residual Waste (Non-Recyclable)',
    badgeVariant: 'neutral',
    description:
      'Composite multi-material packaging bonded from plastic film, aluminum foil, and adhesives. Currently impossible to separate economically in standard recycling facilities.',
    commonExamples: [
      'Potato chip and snack bags',
      'Candy and chocolate bar wrappers',
      'Metallic energy bar pouches',
      'Single-serve coffee pods (unless explicitly compostable)',
      'Plastic straws and disposable plastic cutlery',
    ],
    preparationInstructions: [
      'Empty all crumbs and food remnants.',
      'Dispose of in your household residual waste container.',
      'Consider buying in bulk or choosing paper-packaged snacks to reduce generation.',
    ],
    disposalMethod: {
      binName: 'Black / Dark Grey Residual Trash Bin',
      binColor: '#6B7280',
      instructions: 'Dispose in ordinary household residual trash destined for sanitary landfill or Waste-to-Energy incineration.',
    },
    avoidMixing: [
      'Do not place in plastic recycling bins; the thin metallic coating contaminates pure plastic melting batches.',
    ],
    environmentalBenefits:
      'Proper containment in residual trash prevents lightweight litter from blowing into rivers and ocean ecosystems.',
    iconName: 'Trash2',
  },
  {
    id: 'trash-hazardous-batteries',
    name: 'Batteries & Hazardous Electronics',
    category: 'hazardous',
    recyclingPossibility: 'Special Disposal Required',
    badgeVariant: 'danger',
    description:
      'Lithium-ion, alkaline, nickel-cadmium, and lead-acid batteries. Contain toxic heavy metals and reactive chemicals. NEVER put in ordinary trash or recycling!',
    commonExamples: [
      'Rechargeable lithium-ion batteries (phones, laptops, power tools, vapes)',
      'AA, AAA, 9V alkaline household batteries',
      'Button cell watch batteries',
      'Damaged or swollen battery packs',
      'Broken smartphone chargers and e-waste cables',
    ],
    preparationInstructions: [
      'Tape battery terminals with clear packing tape or electrical tape to prevent short-circuiting during transit.',
      'Store in a cool, non-conductive cardboard box or plastic container.',
      'Never puncture, disassemble, or crush batteries.',
    ],
    disposalMethod: {
      binName: 'Battery Drop-off Depot / Household Hazardous Waste Facility',
      binColor: '#EF4444',
      instructions: 'Drop off at hardware stores (Home Depot, Best Buy), battery recycling kiosks, or local hazardous waste events.',
    },
    avoidMixing: [
      'STRICTLY FORBIDDEN in regular curbside trash or recycling carts! Punctured lithium batteries ignite instantly upon compactor compression.',
    ],
    environmentalBenefits:
      'Safe recovery of critical rare materials including lithium, cobalt, nickel, and copper while preventing groundwater heavy metal contamination.',
    safetyWarnings:
      'CRITICAL FIRE DANGER: Lithium batteries cause hundreds of catastrophic waste truck and sorting facility fires each year. Never discard in curbside bins!',
    iconName: 'BatteryCharging',
  },
]
