import numpy as np
import pandas as pd

np.random.seed(42)

# (1) Import data

counties = [
    "Alameda", "Alpine", "Amador", "Butte", "Calaveras", "Colusa", "Contra Costa", "Del Norte", "El Dorado",
    "Fresno", "Glenn", "Humboldt", "Imperial", "Inyo", "Kern", "Kings", "Lake", "Lassen", "Los Angeles",
    "Madera", "Marin", "Mendocino", "Merced", "Modoc", "Mono", "Monterey", "Napa", "Nevada", "Orange",
    "Placer", "Plumas", "Riverside", "Sacramento", "San Benito", "San Bernardino", "San Diego",
    "San Francisco", "San Joaquin", "San Luis Obispo", "San Mateo", "Santa Barbara", "Santa Clara",
    "Santa Cruz", "Shasta", "Sierra", "Siskiyou", "Solano", "Sonoma", "Stanislaus", "Sutter", "Tehama",
    "Tulare", "Tuolumne", "Ventura", "Yolo", "Yuba"
]

county_acreage_distribution = {
    "Alameda": {'Moderate': 22797, 'High': 24805, 'Very High': 2845},
    "Alpine":  {'Moderate': 5679,  'High': 14495,  'Very High': 14159},
    "Amador":  {'Moderate': 605, 'High': 5121, 'Very High': 2462},
    "Butte":   {'Moderate': 36692, 'High': 21560, 'Very High': 14531},
    "Calaveras": {'Moderate': 338, 'High': 1843, 'Very High': 368},
    "Colusa": {"Moderate": 1787, "High": 6280, "Very High": 0},
    "Contra Costa": {"Moderate": 26522, "High": 30112, "Very High": 16014},
    "Del Norte": {"Moderate": 2436, "High": 69, "Very High": 0},
    "El Dorado": {"Moderate": 3336, "High": 2642, "Very High": 10128},
    "Fresno": {"Moderate": 22466, "High": 9037, "Very High": 0},
    "Glenn": {"Moderate": 4406, "High": 15802, "Very High": 0},
    "Humboldt": {"Moderate": 27155, "High": 2671, "Very High": 878},
    "Imperial": {"Moderate": 340610, "High": 241, "Very High": 0},
    "Inyo": {"Moderate": 71202, "High": 44217, "Very High": 2168},
    "Kern": {"Moderate": 604731, "High": 145012, "Very High": 3407},
    "Kings": {"Moderate": 30104, "High": 28388, "Very High": 0},
    "Lake": {"Moderate": 7564, "High": 7516, "Very High": 15538},
    "Lassen": {"Moderate": 26010, "High": 13148, "Very High": 3563},
    "Los Angeles": {"Moderate": 311362, "High": 56607, "Very High": 314642},
    "Madera": {"Moderate": 32600, "High": 1332, "Very High": 0},
    "Marin": {"Moderate": 15879, "High": 12612, "Very High": 2640},
    "Mendocino": {"Moderate": 14713, "High": 7515, "Very High": 9967},
    "Merced": {"Moderate": 96748, "High": 1806, "Very High": 0},
    "Modoc": {"Moderate": 30052, "High": 32289, "Very High": 2977},
    "Mono": {"Moderate": 9518, "High": 2024, "Very High": 2908},
    "Monterey": {"Moderate": 53938, "High": 15649, "Very High": 8854},
    "Napa": {"Moderate": 11731, "High": 9160, "Very High": 5040},
    "Nevada": {"Moderate": 293, "High": 4047, "Very High": 20987},
    "Orange": {"Moderate": 10057, "High": 21087, "Very High": 64999},
    "Placer": {"Moderate": 38755, "High": 9350, "Very High": 4185},
    "Plumas": {"Moderate": 488, "High": 1529, "Very High": 4636},
    "Riverside": {"Moderate": 435120, "High": 36916, "Very High": 157617},
    "Sacramento": {"Moderate": 77884, "High": 2191, "Very High": 1267},
    "San Benito": {"Moderate": 15847, "High": 1027, "Very High": 372},
    "San Bernardino": {"Moderate": 1495773, "High": 150157, "Very High": 117516},
    "San Diego": {"Moderate": 282113, "High": 53517, "Very High": 157339},
    "San Francisco": {"Moderate": 218, "High": 117, "Very High": 166},
    "San Joaquin": {"Moderate": 8903, "High": 5299, "Very High": 0},
    "San Luis Obispo": {"Moderate": 31206, "High": 29372, "Very High": 12910},
    "San Mateo": {"Moderate": 14086, "High": 15240, "Very High": 7097},
    "Santa Barbara": {"Moderate": 13833, "High": 8994, "Very High": 8714},
    "Santa Clara": {"Moderate": 25480, "High": 29504, "Very High": 14926},
    "Santa Cruz": {"Moderate": 8318, "High": 8462, "Very High": 529},
    "Shasta": {"Moderate": 15288, "High": 25241, "Very High": 30493},
    "Sierra": {"Moderate": 544, "High": 944, "Very High": 388},
    "Siskiyou": {"Moderate": 9954, "High": 41533, "Very High": 16446},
    "Solano": {"Moderate": 24804, "High": 79211, "Very High": 821},
    "Sonoma": {"Moderate": 58919, "High": 13604, "Very High": 9412},
    "Stanislaus": {"Moderate": 46166, "High": 16477, "Very High": 733},
    "Sutter": {"Moderate": 11905, "High": 28808, "Very High": 18771},
    "Tehama": {"Moderate": 19575, "High": 21723, "Very High": 989},
    "Tulare": {"Moderate": 22859, "High": 3260, "Very High": 487},
    "Tuolumne": {"Moderate": 0, "High": 744, "Very High": 1270},
    "Ventura": {"Moderate": 12370, "High": 17583, "Very High": 88810},
    "Yolo": {"Moderate": 3488, "High": 34041, "Very High": 1114},
    "Yuba": {"Moderate": 15611, "High": 1592, "Very High": 0}
    }

county_property_values = {
    "Alameda": 1105150,
    "Alpine": 484530,
    "Amador": 439520,
    "Butte": 420160,
    "Calaveras": 459270,
    "Colusa": 389930,
    "Contra Costa": 858620,
    "Del Norte": 332240,
    "El Dorado": 713650,
    "Fresno": 419380,
    "Glenn": 351780,
    "Humboldt": 486510,
    "Imperial": 355230,
    "Inyo": 351780,
    "Kern": 355240,
    "Kings": 370760,
    "Lake": 350020,
    "Lassen": 269760,
    "Los Angeles": 874980,
    "Madera": 424120,
    "Marin": 1565090,
    "Mariposa": 372160,
    "Mendocino": 530580,
    "Merced": 448700,
    "Modoc": 220380,
    "Mono": 534640,
    "Monterey": 824420,
    "Napa": 869480,
    "Nevada": 672790,
    "Orange": 1015100,
    "Placer": 695390,
    "Plumas": 340350,
    "Riverside": 605280,
    "Sacramento": 550920,
    "San Benito": 779480,
    "San Bernardino": 541700,
    "San Diego": 916050,
    "San Francisco": 1395150,
    "San Joaquin": 547070,
    "San Luis Obispo": 803990,
    "San Mateo": 1544810,
    "Santa Barbara": 812960,
    "Santa Clara": 1476980,
    "Santa Cruz": 1036420,
    "Shasta": 369810,
    "Sierra": 347310,
    "Siskiyou": 295750,
    "Solano": 608090,
    "Sonoma": 812130,
    "Stanislaus": 477780,
    "Sutter": 410790,
    "Tehama": 335570,
    "Trinity": 342010,
    "Tulare": 357730,
    "Tuolumne": 422260,
    "Ventura": 845370,
    "Yolo": 629450,
    "Yuba": 396890
}

burn_prob_ranges = {
    "Very High": (0.01, 0.02),
    "High": (0.005, 0.01),
    "Moderate": (0.0005, 0.005)
}
total_homes = 10000

# Insurers
insurers = ["A", "B", "C", "D", "E", "F", "G", "H"]
scaled_shares = [28.06, 21.00, 9.17, 9.17, 8.60, 8.18, 8.18, 7.63]
scaled_weights = np.array(scaled_shares) / sum(scaled_shares)

# (2) Convert Acreage to Homes, to randomized burn risk, to randomized property value and expected loss

all_homes = []
home_id = 1
total_acreage = sum(sum(risks.values()) for risks in county_acreage_distribution.values())

for county in counties:
    county_risks = county_acreage_distribution[county]
    mean_value = county_property_values[county]
    county_total_acreage = sum(county_risks.values())
    county_home_count = int((county_total_acreage / total_acreage) * total_homes)

    for risk_level, acres in county_risks.items():
        risk_home_count = int((acres / county_total_acreage) * county_home_count)
        for _ in range(risk_home_count):
            burn_prob = np.random.uniform(*burn_prob_ranges[risk_level])
            prop_val = np.random.normal(loc=mean_value, scale=50000)
            expected_loss = burn_prob * prop_val
            all_homes.append([home_id, county, risk_level, burn_prob, prop_val, expected_loss])
            home_id += 1

homes_df = pd.DataFrame(all_homes, columns=["HomeID", "County", "RiskLevel", "BurnProb", "PropValue", "ExpectedLoss"])

# (3) Allocate to Insurers Based on Market Share and Equal Expected Loss (Per County)

home_to_insurer = {}
remaining_homes = homes_df.copy()

for county in counties:
    # first make sure that they get proportional to market share, with rounding caveat
    county_homes = remaining_homes[remaining_homes["County"] == county]
    n_county_homes = len(county_homes)
    insurer_allocations = (scaled_weights * n_county_homes).astype(int)
    while insurer_allocations.sum() < n_county_homes:
        insurer_allocations[np.argmin(insurer_allocations)] += 1

    county_remaining = county_homes.copy()
    buckets = {ins: [] for ins in insurers}

    from itertools import cycle

    county_remaining = county_remaining.sample(frac=1, random_state=42)  # Shuffle first
    # Sort by expected loss (highest risk first)
    county_remaining = county_remaining.sort_values("ExpectedLoss", ascending=False).copy()
    home_ids = county_remaining["HomeID"].tolist()

    # Initialize buckets and loss trackers
    buckets = {ins: [] for ins in insurers}
    remaining_quota = dict(zip(insurers, insurer_allocations))
    total_loss = {ins: 0 for ins in insurers}

    # Sort homes by expected loss descending (high-risk first)
    county_remaining = county_remaining.sample(frac=1, random_state=42)
    county_remaining = county_remaining.sort_values("ExpectedLoss", ascending=False).copy()

    # Loop through each home
    for _, row in county_remaining.iterrows():
        hid = row["HomeID"]
        loss = row["ExpectedLoss"]

        # Find the eligible insurer with the lowest total expected loss
        eligible = [ins for ins in insurers if remaining_quota[ins] > 0]
        best = min(eligible, key=lambda ins: total_loss[ins] / (len(buckets[ins]) + 1e-6))

        # Assign home
        buckets[best].append(hid)
        total_loss[best] += loss
        remaining_quota[best] -= 1

for insurer, home_ids in buckets.items():
    for hid in home_ids:
        home_to_insurer[hid] = insurer
homes_df["Insurer"] = homes_df["HomeID"].map(home_to_insurer)

# Rebalance average expected loss per insurer via pairwise home swaps

tolerance = 100
max_iterations = 1000
iteration = 0

while iteration < max_iterations:
    # Compute average expected loss per insurer
    avg_loss = homes_df.groupby("Insurer")["ExpectedLoss"].mean()
    high = avg_loss.idxmax()
    low = avg_loss.idxmin()
    diff = avg_loss[high] - avg_loss[low]

    if diff <= tolerance:
        print(f"\n✅ Balanced within {tolerance} after {iteration} iterations.")
        break

    # Get the riskiest home from the overburdened insurer
    high_home = homes_df[homes_df["Insurer"] == high]\
        .sort_values("ExpectedLoss", ascending=False).iloc[0]
    low_home = homes_df[homes_df["Insurer"] == low]\
        .sort_values("ExpectedLoss", ascending=True).iloc[0]

    high_id = high_home["HomeID"]
    low_id = low_home["HomeID"]

    # Only swap if homes are distinct and beneficial
    if high_id == low_id or high_home["ExpectedLoss"] <= low_home["ExpectedLoss"]:
        print("⛔ No beneficial swap found. Exiting.")
        break

    # 🛠 Use TEMP placeholder to prevent overwrite
    homes_df.loc[homes_df["HomeID"] == high_id, "Insurer"] = "TEMP"
    homes_df.loc[homes_df["HomeID"] == low_id, "Insurer"] = high
    homes_df.loc[homes_df["HomeID"] == high_id]()

import matplotlib.pyplot as plt

homes_df.groupby("Insurer")["ExpectedLoss"].mean().plot(kind="bar")
plt.ylabel("Avg Expected Loss per Home")
plt.title("Expected Loss by Insurer")
plt.show()

# (4) Summary Table + Export to Excel

allocation_summary = homes_df.groupby(["County", "RiskLevel", "Insurer"]).size().reset_index(name="NumHomes")
homes_df.to_excel("homes_data.xlsx", index=False)
allocation_summary.to_excel("allocation_summary.xlsx", index=False)

# (5) Lookup Function

home_insurer_lookup = homes_df.set_index("HomeID")["Insurer"].to_dict()
def get_insurer(home_id):
    return home_insurer_lookup.get(home_id, "Home ID not found")

# (6) Native Home Lookup

def home_lookup(home_id):
    try:
        row = homes_df.loc[homes_df["HomeID"] == home_id].iloc[0]
        print(f"Home ID: {home_id}")
        print(f"County: {row['County']}")
        print(f"Risk Level: {row['RiskLevel']}")
        print(f"Burn Probability: {row['BurnProb']:.4f}")
        print(f"Property Value: ${row['PropValue']:,.2f}")
        print(f"Expected Loss: ${row['ExpectedLoss']:,.2f}")
        print(f"Insurance Company: {row['Insurer']}")
    except IndexError:
        print("Home ID not found. Please enter a valid ID between 1 and 10,000.")

while True:
    try:
        user_input = input("Enter a Home ID (1–10000), or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
        home_id = int(user_input)
        home_lookup(home_id)
    except ValueError:
        print("Please enter a valid integer or 'q' to quit.")