import pandas as pd

# 1. Insurer market shares (scaled to sum to 1)
insurers = {
    'A': 28.06,
    'B': 21.00,
    'C': 9.17,
    'D': 9.17,
    'E': 8.60,
    'F': 8.18,
    'G': 8.18,
    'H': 7.63
}
total_share = sum(insurers.values())
insurer_shares = {k: v / total_share for k, v in insurers.items()}

# 2. County-level data (from image): Combined LRA+SRA acres
county_data = {
    'Alameda': {'Moderate': 22797, 'High': 24805, 'Very High': 2845, 'Total': 50446},
    'Alpine':  {'Moderate': 5679,  'High': 14495,  'Very High': 14159,  'Total': 34333},
    'Amador':  {'Moderate': 605, 'High': 5121, 'Very High': 2462, 'Total': 8188},
    'Butte':   {'Moderate': 36692, 'High': 21560, 'Very High': 14531, 'Total': 72783},
    'Calaveras': {'Moderate': 338, 'High': 1843, 'Very High': 368, 'Total': 2550},
    'Colusa': {'Moderate': 1787, 'High': 6280, 'Very High': 0, 'Total': 8067},
    'Contra Costa': {'Moderate': 26522, 'High': 30112, 'Very High': 16014, 'Total': 72648},
    'Del Norte': {'Moderate': 2436, 'High': 69, 'Very High': 0, 'Total': 2505},
    'El Dorado': {'Moderate': 3336, 'High': 2642, 'Very High': 10128, 'Total': 16107},
    'Fresno': {'Moderate': 22466, 'High': 9037, 'Very High': 0, 'Total': 31502},
    'Glenn': {'Moderate': 4406, 'High': 15802, 'Very High': 0, 'Total': 20208},
    'Humboldt': {'Moderate': 27155, 'High': 2671, 'Very High': 878, 'Total': 30703},
    'Imperial': {'Moderate': 340610, 'High': 241, 'Very High': 0, 'Total': 340851},
    'Inyo': {'Moderate': 71202, 'High': 44217, 'Very High': 2168, 'Total': 117587},
    'Kern': {'Moderate': 604731, 'High': 145012, 'Very High': 3407, 'Total': 753150},
    'Kings': {'Moderate': 30104, 'High': 28388, 'Very High': 0, 'Total': 58492},
    'Lake': {'Moderate': 7564, 'High': 7516, 'Very High': 15538, 'Total': 30619},
    'Lassen': {'Moderate': 26010, 'High': 13148, 'Very High': 3563, 'Total': 42721},
    'Los Angeles': {'Moderate': 312362, 'High': 56607, 'Very High': 314642, 'Total': 683610},
    'Madera': {'Moderate': 32600, 'High': 1332, 'Very High': 0, 'Total': 33932},
    'Marin': {'Moderate': 15879, 'High': 12612, 'Very High': 2640, 'Total': 31131},
    'Mendocino': {'Moderate': 14713, 'High': 7515, 'Very High': 9967, 'Total': 32194},
    'Merced': {'Moderate': 96748, 'High': 1806, 'Very High': 0, 'Total': 98554},
    'Modoc': {'Moderate': 30052, 'High': 32289, 'Very High': 2977, 'Total': 65319},
    'Mono': {'Moderate': 9518, 'High': 2024, 'Very High': 2908, 'Total': 14450},
    'Monterey': {'Moderate': 53938, 'High': 15649, 'Very High': 8854, 'Total': 78442},
    'Napa': {'Moderate': 11731, 'High': 9160, 'Very High': 5040, 'Total': 25932},
    'Nevada': {'Moderate': 293, 'High': 4047, 'Very High': 20987, 'Total': 25328},
    'Orange': {'Moderate': 10057, 'High': 21087, 'Very High': 64999, 'Total': 96142},
    'Placer': {'Moderate': 38755, 'High': 9350, 'Very High': 4185, 'Total': 52290},
    'Plumas': {'Moderate': 488, 'High': 1529, 'Very High': 4636, 'Total': 6653},
    'Riverside': {'Moderate': 435120, 'High': 36916, 'Very High': 157617, 'Total': 629652},
    'Sacramento': {'Moderate': 77884, 'High': 2191, 'Very High': 1267, 'Total': 81342},
    'San Benito': {'Moderate': 15847, 'High': 1027, 'Very High': 372, 'Total': 17247},
    'San Bernardino': {'Moderate': 1495773, 'High': 150157, 'Very High': 117516, 'Total': 1763446},
    'San Diego': {'Moderate': 282113, 'High': 53517, 'Very High': 157339, 'Total': 492969},
    'San Francisco': {'Moderate': 218, 'High': 117, 'Very High': 166, 'Total': 502},
    'San Joaquin': {'Moderate': 8903, 'High': 5299, 'Very High': 0, 'Total': 14202},
    'San Luis Obispo': {'Moderate': 31206, 'High': 29372, 'Very High': 12910, 'Total': 73488},
    'San Mateo': {'Moderate': 14086, 'High': 15240, 'Very High': 7097, 'Total': 36423},
    'Santa Barbara': {'Moderate': 13833, 'High': 8994, 'Very High': 8714, 'Total': 31541},
    'Santa Clara': {'Moderate': 25480, 'High': 29504, 'Very High': 14926, 'Total': 69909},
    'Santa Cruz': {'Moderate': 8318, 'High': 8462, 'Very High': 529, 'Total': 17309},
    'Shasta': {'Moderate': 15288, 'High': 25241, 'Very High': 30493, 'Total': 71022},
    'Sierra': {'Moderate': 544, 'High': 944, 'Very High': 388, 'Total': 1875},
    'Siskiyou': {'Moderate': 9954, 'High': 41533, 'Very High': 16446, 'Total': 67932},
    'Solano': {'Moderate': 24804, 'High': 79211, 'Very High': 821, 'Total': 104835},
    'Sonoma': {'Moderate': 58919, 'High': 13604, 'Very High': 9412, 'Total': 81935},
    'Stanislaus': {'Moderate': 46166, 'High': 16477, 'Very High': 733, 'Total': 63376},
    'Sutter': {'Moderate': 11905, 'High': 28808, 'Very High': 18771, 'Total': 59485},
    'Tehama': {'Moderate': 19575, 'High': 21723, 'Very High': 989, 'Total': 42287},
    'Tulare': {'Moderate': 22859, 'High': 3260, 'Very High': 487, 'Total': 26606},
    'Tuolumne': {'Moderate': 0, 'High': 744, 'Very High': 1270, 'Total': 2015},
    'Ventura': {'Moderate': 12370, 'High': 17583, 'Very High': 88810, 'Total': 118763},
    'Yolo': {'Moderate': 3488, 'High': 34041, 'Very High': 1114, 'Total': 38643},
    'Yuba': {'Moderate': 15611, 'High': 1592, 'Very High': 0, 'Total': 17204}
}

# --- Step 3: Choose total number of homes to allocate ---
n_homes = 100000

# --- Step 4: Distribute homes to counties based on their share of total LRA acreage ---
total_acres = sum(c['Total'] for c in county_data.values())
county_homes = {
    county: round((cdata['Total'] / total_acres) * n_homes)
    for county, cdata in county_data.items()
}

# --- Step 4: Allocate homes within each county by risk tier ---
allocations = []
for county, total_h in county_homes.items():
    risks = county_data[county]
    total_acres_county = risks['Total']

    risk_tiers = {
        'Moderate': risks['Moderate'],
        'High': risks['High'],
        'Severe': risks['Very High']
    }

    for risk, acres in risk_tiers.items():
        homes_in_risk = round((acres / total_acres_county) * total_h)
        for insurer, share in insurer_shares.items():
            allocated = round(homes_in_risk * share)
            allocations.append({
                'County': county,
                'Risk Level': risk,
                'Insurer': insurer,
                'Homes Allocated': allocated
            })

# --- Step 6: Aggregate result like your screenshot ---
df = pd.DataFrame(allocations)
summary = df.groupby(['Insurer', 'Risk Level'])['Homes Allocated'].sum().reset_index()
summary = summary.sort_values(by=['Insurer', 'Risk Level'])

print(summary)
summary.to_excel("/Users/chelseahu/Documents/homes_allocated_summary.xlsx", index=False)