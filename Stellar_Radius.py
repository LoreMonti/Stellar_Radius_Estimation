# ==========================================================
# Stellar Radius Estimation from Multi-band Photometry
# and Infrared Monte Carlo Sampling
#
# Author: Lorenzo Monti
# ==========================================================


# --- Standard library imports ---
import csv
from pathlib import Path


# --- Third-party imports ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_DIR = Path("/Users/lorenzo/Desktop/GitHub/Star")

DATA_FILE    = PROJECT_DIR / "Data" / "Data.csv"
BC_FILE      = PROJECT_DIR / "Data" / "output.file.all"

OUTPUT_DIR   = PROJECT_DIR / "Plots"
RESULTS_FILE = PROJECT_DIR / "Result" / "Radius.csv"
OUTLIER_FILE = PROJECT_DIR / "Result" / "Radius_outliers.csv"

N_SAMPLES   = 10000
RANDOM_SEED = 42

T_SUN     = 5771.8
M_BOL_SUN = 4.75


# ==========================================================
# DATA STRUCTURE
# ==========================================================

BANDS = {
    "BT": {"mag": "mBT", "e_mag": "e_mBT", "bc": "BC_BT", "e_bc": "e_BC_BT"},
    "VT": {"mag": "mVT", "e_mag": "e_mVT", "bc": "BC_VT", "e_bc": "e_BC_VT"},
    "G":  {"mag": "mG" , "e_mag": "e_mG" , "bc": "BC_G" , "e_bc": "e_BC_G" },
    "BP": {"mag": "mBP", "e_mag": "e_mBP", "bc": "BC_BP", "e_bc": "e_BC_BP"},
    "RP": {"mag": "mRP", "e_mag": "e_mRP", "bc": "BC_RP", "e_bc": "e_BC_RP"},
    "J":  {"mag": "mJ" , "e_mag": "e_mJ" , "bc": "BC_J" , "e_bc": "e_BC_J" },
    "H":  {"mag": "mH" , "e_mag": "e_mH" , "bc": "BC_H" , "e_bc": "e_BC_H" },
    "K":  {"mag": "mK" , "e_mag": "e_mK" , "bc": "BC_K" , "e_bc": "e_BC_K" },
}

IR_BANDS = ["J", "H", "K"]

COLUMN_NAMES = [
    "ID",
    "logg", "e_logg",
    "Fe_H", "e_Fe_H",
    "Teff", "e_Teff",
    "B_V",
    "Plx", "e_Plx",
    "mBT", "e_mBT",
    "mVT", "e_mVT",
    "mG", "e_mG",
    "mBP", "e_mBP",
    "mRP", "e_mRP",
    "mJ", "e_mJ",
    "mH", "e_mH",
    "mK", "e_mK",
    "BC_BT", "e_BC_BT",
    "BC_VT", "e_BC_VT",
    "BC_G", "e_BC_G",
    "BC_BP", "e_BC_BP",
    "BC_RP", "e_BC_RP",
    "BC_J", "e_BC_J",
    "BC_H", "e_BC_H",
    "BC_K", "e_BC_K",
    "R_A", "e_R_A",
    "R_b", "e_R_b",
]


# ==========================================================
# DATA LOADING AND VALIDATION
# ==========================================================
def load_stellar_catalogue(file_path):
    """
    Load the stellar catalogue.

    The input catalogue is expected to contain one row per star.
    Numeric columns are converted explicitly to floating-point values.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input catalogue not found: {file_path}")

    data = pd.read_csv(file_path, sep=",")

    if len(data.columns) == len(COLUMN_NAMES):
        data.columns = COLUMN_NAMES

    for column in data.columns:
        if column != "ID":
            data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def validate_catalogue(data):
    """
    Remove stars with invalid fundamental quantities.

    Stars with non-positive parallax, effective temperature, or reference
    radius cannot be used for physically meaningful radius estimates.
    """

    initial_size = len(data)

    valid_mask = (
        np.isfinite(data["Plx"])
        & np.isfinite(data["e_Plx"])
        & np.isfinite(data["Teff"])
        & np.isfinite(data["e_Teff"])
        & np.isfinite(data["R_A"])
        & np.isfinite(data["e_R_A"])
        & (data["Plx"] > 0.0)
        & (data["e_Plx"] > 0.0)
        & (data["Teff"] > 0.0)
        & (data["e_Teff"] > 0.0)
        & (data["R_A"] > 0.0)
        & (data["e_R_A"] > 0.0)
    )

    cleaned_data = data.loc[valid_mask].reset_index(drop=True)

    removed = initial_size - len(cleaned_data)

    print(f"Initial number of stars: {initial_size}")
    print(f"Removed invalid stars: {removed}")
    print(f"Valid stars: {len(cleaned_data)}")

    return cleaned_data


# ==========================================================
# BASIC PHYSICAL COMPUTATIONS
# ==========================================================
def compute_distance(parallax_mas, parallax_error_mas):
    """
    Compute distance from parallax.

    Parameters
    ----------
    parallax_mas : array-like
        Parallax in milliarcseconds.

    parallax_error_mas : array-like
        Parallax uncertainty in milliarcseconds.

    Returns
    -------
    distance_pc : array-like
        Distance in parsec.

    distance_error_pc : array-like
        Propagated distance uncertainty.
    """

    parallax_arcsec = parallax_mas * 1.0e-3
    parallax_error_arcsec = parallax_error_mas * 1.0e-3

    distance_pc = 1.0 / parallax_arcsec
    distance_error_pc = parallax_error_arcsec / parallax_arcsec**2

    return distance_pc, distance_error_pc


def compute_bolometric_magnitude(
    apparent_mag,
    apparent_mag_error,
    bolometric_correction,
    bolometric_correction_error,
    distance,
    distance_error,
):
    """
    Compute apparent and absolute bolometric magnitude.

    m_bol = m_band + BC_band

    M_bol = m_bol - 5 log10(d) + 5
    """

    m_bol = apparent_mag + bolometric_correction

    m_bol_error = np.sqrt(
        apparent_mag_error**2
        + bolometric_correction_error**2
    )

    M_bol = m_bol - 5.0 * np.log10(distance) + 5.0

    M_bol_error = np.sqrt(
        m_bol_error**2
        + (5.0 * distance_error / (distance * np.log(10.0)))**2
    )

    return m_bol, m_bol_error, M_bol, M_bol_error


def compute_radius(teff, teff_error, M_bol, M_bol_error):
    """
    Compute stellar radius from effective temperature and bolometric magnitude.

    R / R_sun = (T_sun / T_eff)^2
                * 10^((M_bol_sun - M_bol) / 5)
    """

    radius = (
        (T_SUN / teff) ** 2
        * 10.0 ** ((M_BOL_SUN - M_bol) / 5.0)
    )

    dR_dTeff = -2.0 * radius / teff

    dR_dMbol = -np.log(10.0) / 5.0 * radius

    radius_error = np.sqrt(
        (dR_dTeff * teff_error) ** 2
        + (dR_dMbol * M_bol_error) ** 2
    )

    return radius, radius_error


# ==========================================================
# DETERMINISTIC MULTI-BAND ANALYSIS
# ==========================================================
def compute_all_band_radii(data):
    """
    Compute deterministic radius estimates for all available bands.

    This stage provides a broad multi-band diagnostic estimate.
    It is not intended to replace the final infrared Monte Carlo estimate.
    """

    data = data.copy()

    distance, distance_error = compute_distance(
        data["Plx"],
        data["e_Plx"],
    )

    data["distance"] = distance
    data["e_distance"] = distance_error

    for band, columns in BANDS.items():

        _, _, M_bol, M_bol_error = compute_bolometric_magnitude(
            apparent_mag=data[columns["mag"]],
            apparent_mag_error=data[columns["e_mag"]],
            bolometric_correction=data[columns["bc"]],
            bolometric_correction_error=data[columns["e_bc"]],
            distance=distance,
            distance_error=distance_error,
        )

        radius, radius_error = compute_radius(
            teff=data["Teff"],
            teff_error=data["e_Teff"],
            M_bol=M_bol,
            M_bol_error=M_bol_error,
        )

        data[f"Mbol_from_{band}"] = M_bol
        data[f"e_Mbol_from_{band}"] = M_bol_error
        data[f"R_{band}_det"] = radius
        data[f"e_R_{band}_det"] = radius_error

    return data


def compute_weighted_multiband_radius(data):
    """
    Compute weighted mean radius from all deterministic band estimates.

    The weights are defined as inverse variances:

        w_i = 1 / sigma_i^2

    This is a diagnostic estimate only, because different bands share
    common uncertainties from Teff and parallax.
    """

    data = data.copy()

    radius_columns = [f"R_{band}_det" for band in BANDS]
    error_columns = [f"e_R_{band}_det" for band in BANDS]

    radii = data[radius_columns].to_numpy(dtype=float)
    errors = data[error_columns].to_numpy(dtype=float)

    valid = np.isfinite(radii) & np.isfinite(errors) & (errors > 0.0)

    weights = np.zeros_like(errors)
    weights[valid] = 1.0 / errors[valid] ** 2

    weighted_sum = np.sum(weights * radii, axis=1)
    weight_sum = np.sum(weights, axis=1)

    data["R_all_det"] = weighted_sum / weight_sum
    data["e_R_all_det"] = np.sqrt(1.0 / weight_sum)

    data.loc[weight_sum <= 0.0, "R_all_det"] = np.nan
    data.loc[weight_sum <= 0.0, "e_R_all_det"] = np.nan

    data["Z_all_det"] = compute_z_score(
        radius=data["R_all_det"],
        radius_error=data["e_R_all_det"],
        reference_radius=data["R_A"],
        reference_error=data["e_R_A"],
    )

    return data


def compute_z_score(radius, radius_error, reference_radius, reference_error):
    """
    Compute the Z-score between a radius estimate and the reference radius.
    """

    return (
        (radius - reference_radius)
        / np.sqrt(radius_error**2 + reference_error**2)
    )


# ==========================================================
# INFRARED BOLOMETRIC CORRECTIONS
# ==========================================================
def read_ir_bolometric_corrections(file_path):
    """
    Read infrared bolometric corrections from the Monte Carlo BC output file.

    Expected format:

        ID log(g) [Fe/H] Teff E(B-V) BC_1 BC_2 BC_3 BC_4 BC_5

    Current mapping:

        BC_1 -> J-band bolometric correction
        BC_2 -> H-band bolometric correction
        BC_3 -> K-band bolometric correction
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"BC file not found: {file_path}")

    corrections = pd.read_csv(
        file_path,
        sep=r"\s+",
        engine="python",
    )

    expected_columns = [
        "ID",
        "log(g)",
        "[Fe/H]",
        "Teff",
        "E(B-V)",
        "BC_1",
        "BC_2",
        "BC_3",
        "BC_4",
        "BC_5",
    ]

    missing_columns = [
        column for column in expected_columns
        if column not in corrections.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in BC file: {missing_columns}\n"
            f"Available columns are: {list(corrections.columns)}"
        )

    corrections = corrections.rename(
        columns={
            "Teff": "Teff_MC",
            "BC_1": "BC_J",
            "BC_2": "BC_H",
            "BC_3": "BC_K",
            "BC_4": "BC_G",
            "BC_5": "BC_aux",
        }
    )

    numeric_columns = [
        "log(g)",
        "[Fe/H]",
        "Teff_MC",
        "E(B-V)",
        "BC_J",
        "BC_H",
        "BC_K",
        "BC_G",
        "BC_aux",
    ]

    for column in numeric_columns:
        corrections[column] = pd.to_numeric(corrections[column], errors="coerce")

    corrections = corrections.dropna(
        subset=["Teff_MC", "BC_J", "BC_H", "BC_K"]
    ).reset_index(drop=True)

    print(f"BC file found: {file_path}")
    print(f"Number of parsed BC samples: {len(corrections)}")

    return corrections


def reshape_ir_bolometric_corrections(corrections, data, n_samples):
    """
    Reshape infrared bolometric corrections into one Monte Carlo block per star.

    The BC file is assumed to be ordered by star, with n_samples consecutive
    samples for each object.
    """

    n_stars = len(data)
    expected_size = n_stars * n_samples

    if len(corrections) < expected_size:
        raise ValueError(
            f"Not enough BC samples. Expected {expected_size}, "
            f"but found {len(corrections)}."
        )

    corrections = corrections.iloc[:expected_size].copy()

    bc_ids = corrections["ID"].to_numpy().reshape(n_stars, n_samples)
    catalogue_ids = data["ID"].to_numpy()

    for i in range(n_stars):
        unique_ids = np.unique(bc_ids[i])

        if len(unique_ids) != 1:
            raise ValueError(
                f"BC samples for catalogue row {i} contain multiple IDs: "
                f"{unique_ids}"
            )

        if str(unique_ids[0]) != str(catalogue_ids[i]):
            raise ValueError(
                f"ID mismatch at row {i}: catalogue ID = {catalogue_ids[i]}, "
                f"BC ID = {unique_ids[0]}"
            )

    bc_samples = {
        "Teff": corrections["Teff_MC"].to_numpy().reshape(n_stars, n_samples),
        "J": corrections["BC_J"].to_numpy().reshape(n_stars, n_samples),
        "H": corrections["BC_H"].to_numpy().reshape(n_stars, n_samples),
        "K": corrections["BC_K"].to_numpy().reshape(n_stars, n_samples),
    }

    return bc_samples


# ==========================================================
# INFRARED MONTE CARLO SAMPLING
# ==========================================================
def sample_ir_observables(data, n_samples, random_seed=None):
    """
    Generate Monte Carlo samples for parallax and infrared magnitudes.
    """

    rng = np.random.default_rng(random_seed)

    samples = {}

    samples["Plx"] = rng.normal(
        loc=data["Plx"].to_numpy()[:, None],
        scale=data["e_Plx"].to_numpy()[:, None],
        size=(len(data), n_samples),
    )

    for band in IR_BANDS:

        columns = BANDS[band]

        samples[f"m{band}"] = rng.normal(
            loc=data[columns["mag"]].to_numpy()[:, None],
            scale=data[columns["e_mag"]].to_numpy()[:, None],
            size=(len(data), n_samples),
        )

    return samples


def compute_distance_from_parallax_samples(parallax_mas):
    """
    Compute distance samples from parallax samples.

    Invalid non-positive parallaxes are converted to NaN.
    """

    distance = np.full_like(parallax_mas, np.nan, dtype=float)

    valid_mask = parallax_mas > 0.0

    distance[valid_mask] = 1.0 / (parallax_mas[valid_mask] * 1.0e-3)

    return distance


def compute_radius_from_ir_samples(teff, apparent_mag, bolometric_correction, distance):
    """
    Compute stellar radius from Monte Carlo infrared samples.
    """

    M_bol = (
        apparent_mag
        + bolometric_correction
        - 5.0 * np.log10(distance)
        + 5.0
    )

    radius = (
        (T_SUN / teff) ** 2
        * 10.0 ** ((M_BOL_SUN - M_bol) / 5.0)
    )

    return radius


def clean_radius_samples(radius_samples, teff_samples):
    """
    Remove non-physical or invalid Monte Carlo samples.

    The original thesis workflow treated Teff = 7500 K as a placeholder
    value, therefore these samples are excluded.
    """

    valid_mask = (
        np.isfinite(radius_samples)
        & np.isfinite(teff_samples)
        & (radius_samples > 0.0)
        & (teff_samples > 0.0)
        & (teff_samples != 7500.0)
    )

    cleaned_samples = []

    for i in range(radius_samples.shape[0]):
        cleaned_samples.append(radius_samples[i, valid_mask[i]])

    return cleaned_samples


def compute_ir_radius_distributions(data, samples, bc_samples):
    """
    Compute Monte Carlo radius distributions for J, H, and K bands.
    """

    n_stars = len(data)

    distance_samples = compute_distance_from_parallax_samples(samples["Plx"])

    radius_samples = {}
    radius_mean = {}
    radius_std = {}
    radius_median = {}
    radius_p16 = {}
    radius_p84 = {}

    for band in IR_BANDS:

        raw_radius = compute_radius_from_ir_samples(
            teff=bc_samples["Teff"],
            apparent_mag=samples[f"m{band}"],
            bolometric_correction=bc_samples[band],
            distance=distance_samples,
        )

        cleaned_radius = clean_radius_samples(
            radius_samples=raw_radius,
            teff_samples=bc_samples["Teff"],
        )

        radius_samples[band] = cleaned_radius

        radius_mean[band] = np.array(
            [
                np.mean(cleaned_radius[i]) if len(cleaned_radius[i]) > 0 else np.nan
                for i in range(n_stars)
            ]
        )

        radius_std[band] = np.array(
            [
                np.std(cleaned_radius[i], ddof=1) if len(cleaned_radius[i]) > 1 else np.nan
                for i in range(n_stars)
            ]
        )

        radius_median[band] = np.array(
            [
                np.median(cleaned_radius[i]) if len(cleaned_radius[i]) > 0 else np.nan
                for i in range(n_stars)
            ]
        )

        radius_p16[band] = np.array(
            [
                np.percentile(cleaned_radius[i], 16) if len(cleaned_radius[i]) > 0 else np.nan
                for i in range(n_stars)
            ]
        )

        radius_p84[band] = np.array(
            [
                np.percentile(cleaned_radius[i], 84) if len(cleaned_radius[i]) > 0 else np.nan
                for i in range(n_stars)
            ]
        )

    return (
        radius_samples,
        radius_mean,
        radius_std,
        radius_median,
        radius_p16,
        radius_p84,
    )


def combine_ir_radius_distributions(radius_samples):
    """
    Combine J, H, and K Monte Carlo radius samples into one IR distribution.

    This combined distribution is used as the final infrared radius estimate.
    """

    n_stars = len(radius_samples["J"])

    combined_samples = []

    for i in range(n_stars):

        samples_i = []

        for band in IR_BANDS:
            if len(radius_samples[band][i]) > 0:
                samples_i.append(radius_samples[band][i])

        if len(samples_i) == 0:
            combined_samples.append(np.array([]))
        else:
            combined_samples.append(np.concatenate(samples_i))

    combined_mean = np.array(
        [
            np.mean(samples) if len(samples) > 0 else np.nan
            for samples in combined_samples
        ]
    )

    combined_std = np.array(
        [
            np.std(samples, ddof=1) if len(samples) > 1 else np.nan
            for samples in combined_samples
        ]
    )

    combined_median = np.array(
        [
            np.median(samples) if len(samples) > 0 else np.nan
            for samples in combined_samples
        ]
    )

    combined_p16 = np.array(
        [
            np.percentile(samples, 16) if len(samples) > 0 else np.nan
            for samples in combined_samples
        ]
    )

    combined_p84 = np.array(
        [
            np.percentile(samples, 84) if len(samples) > 0 else np.nan
            for samples in combined_samples
        ]
    )

    return (
        combined_samples,
        combined_mean,
        combined_std,
        combined_median,
        combined_p16,
        combined_p84,
    )


def add_ir_results_to_catalogue(
    data,
    radius_mean,
    radius_std,
    radius_median,
    radius_p16,
    radius_p84,
    combined_mean,
    combined_std,
    combined_median,
    combined_p16,
    combined_p84,
):
    """
    Add infrared Monte Carlo radius estimates to the catalogue.
    """

    data = data.copy()

    for band in IR_BANDS:

        data[f"R_{band}_MC"] = radius_mean[band]
        data[f"e_R_{band}_MC"] = radius_std[band]

        data[f"R_{band}_MC_median"] = radius_median[band]
        data[f"R_{band}_MC_p16"] = radius_p16[band]
        data[f"R_{band}_MC_p84"] = radius_p84[band]

    data["R_IR"] = combined_mean
    data["e_R_IR"] = combined_std

    data["R_IR_median"] = combined_median
    data["R_IR_p16"] = combined_p16
    data["R_IR_p84"] = combined_p84

    data["Z_IR"] = compute_z_score(
        radius=data["R_IR"],
        radius_error=data["e_R_IR"],
        reference_radius=data["R_A"],
        reference_error=data["e_R_A"],
    )

    data["delta_R_all_det"] = data["R_all_det"] - data["R_A"]
    data["delta_R_IR"] = data["R_IR"] - data["R_A"]

    data["e_Plx_relative"] = 100.0 * data["e_Plx"] / data["Plx"]

    return data


# ==========================================================
# OUTPUT TABLES
# ==========================================================
def save_results_table(data, output_file):
    """
    Save the final stellar radius catalogue.

    The table includes both the deterministic multi-band diagnostic estimate
    and the final infrared Monte Carlo estimate.
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    columns_to_save = [
        "ID",
        "Teff", "e_Teff",
        "Plx", "e_Plx",
        "distance", "e_distance",
        "e_Plx_relative",

        "R_all_det", "e_R_all_det", "Z_all_det",

        "R_J_MC", "e_R_J_MC",
        "R_H_MC", "e_R_H_MC",
        "R_K_MC", "e_R_K_MC",

        "R_IR", "e_R_IR",
        "R_IR_median",
        "R_IR_p16",
        "R_IR_p84",
        "Z_IR",

        "R_A", "e_R_A",
        "delta_R_all_det",
        "delta_R_IR",
    ]

    available_columns = [
        column for column in columns_to_save
        if column in data.columns
    ]

    data[available_columns].to_csv(output_file, index=False)

    print(f"\nResults table saved to: {output_file}")


def save_outlier_table(data, output_file, threshold=1.96):
    """
    Save a diagnostic table for stars failing the infrared Z-test.

    The table is intended to help identify whether outliers are associated
    with parallax uncertainty, temperature, or JHK band inconsistencies.
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    outliers = data[np.abs(data["Z_IR"]) >= threshold].copy()

    outliers["R_K_minus_R_J"] = outliers["R_K_MC"] - outliers["R_J_MC"]

    outliers["e_R_K_minus_R_J"] = np.sqrt(
        outliers["e_R_K_MC"]**2
        + outliers["e_R_J_MC"]**2
    )

    outliers["relative_delta_R_IR"] = (
        100.0 * (outliers["R_IR"] - outliers["R_A"]) / outliers["R_A"]
    )

    columns_to_save = [
        "ID",

        "Z_IR",
        "R_IR", "e_R_IR",
        "R_A", "e_R_A",
        "relative_delta_R_IR",

        "Teff", "e_Teff",
        "Plx", "e_Plx",
        "e_Plx_relative",

        "R_J_MC", "e_R_J_MC",
        "R_H_MC", "e_R_H_MC",
        "R_K_MC", "e_R_K_MC",
        "R_K_minus_R_J",
        "e_R_K_minus_R_J",

        "R_all_det", "e_R_all_det",
        "Z_all_det",
    ]

    available_columns = [
        column for column in columns_to_save
        if column in outliers.columns
    ]

    outliers[available_columns].to_csv(output_file, index=False)

    print(f"Outlier diagnostic table saved to: {output_file}")


# ==========================================================
# DIAGNOSTIC PRINTING
# ==========================================================
def print_global_summary(data):
    """
    Print a global summary of the radius analysis.
    """

    n_stars = len(data)

    n_success_det = np.sum(np.abs(data["Z_all_det"]) < 1.96)
    n_success_ir = np.sum(np.abs(data["Z_IR"]) < 1.96)

    print("\nGlobal summary")
    print("--------------")
    print(f"Number of analysed stars: {n_stars}")

    print(
        f"Deterministic multi-band Z-test success: "
        f"{n_success_det}/{n_stars} "
        f"({100.0 * n_success_det / n_stars:.2f}%)"
    )

    print(
        f"Infrared Monte Carlo Z-test success: "
        f"{n_success_ir}/{n_stars} "
        f"({100.0 * n_success_ir / n_stars:.2f}%)"
    )

    print(
        f"Median |Z_all_det|: "
        f"{np.nanmedian(np.abs(data['Z_all_det'])):.3f}"
    )

    print(
        f"Median |Z_IR|: "
        f"{np.nanmedian(np.abs(data['Z_IR'])):.3f}"
    )


def print_failed_z_tests(data, column="Z_IR", threshold=1.96):
    """
    Print stars that fail a selected Z-test.
    """

    failed = data[np.abs(data[column]) >= threshold].copy()

    if len(failed) == 0:
        print(f"\nNo failed Z-tests found for {column}.")
        return

    failed["R_K_minus_R_J"] = failed["R_K_MC"] - failed["R_J_MC"]

    print(f"\nStars failing {column} test:")

    for _, row in failed.iterrows():
        print(
            f"{row['ID']} | "
            f"{column} = {row[column]:.3f} | "
            f"R_IR = {row['R_IR']:.4f} +- {row['e_R_IR']:.4f} | "
            f"R_A = {row['R_A']:.4f} +- {row['e_R_A']:.4f} | "
            f"e_Plx/Plx = {row['e_Plx_relative']:.2f}% | "
            f"R_K - R_J = {row['R_K_minus_R_J']:.4f}"
        )


def print_star_summary(data, index):
    """
    Print deterministic and infrared radius estimates for one selected star.
    """

    star = data.iloc[index]

    print(f"\nSelected star: {star['ID']}")
    print("--------------------------------")

    print("\nDeterministic multi-band estimates:")

    for band in BANDS:
        print(
            f"{band:>2s} | "
            f"Mbol = {star[f'Mbol_from_{band}']:8.4f} mag | "
            f"R = {star[f'R_{band}_det']:8.4f} +- "
            f"{star[f'e_R_{band}_det']:8.4f} R_sun"
        )

    print(
        f"\nWeighted deterministic radius: "
        f"R_all_det = {star['R_all_det']:.4f} +- "
        f"{star['e_R_all_det']:.4f} R_sun"
    )

    print(f"Z_all_det = {star['Z_all_det']:.4f}")

    print("\nInfrared Monte Carlo estimates:")

    for band in IR_BANDS:
        print(
            f"{band:>2s} | "
            f"R_MC = {star[f'R_{band}_MC']:8.4f} +- "
            f"{star[f'e_R_{band}_MC']:8.4f} R_sun | "
            f"p16-p84 = "
            f"[{star[f'R_{band}_MC_p16']:.4f}, "
            f"{star[f'R_{band}_MC_p84']:.4f}]"
        )

    print(
        f"\nFinal infrared radius: "
        f"R_IR = {star['R_IR']:.4f} +- {star['e_R_IR']:.4f} R_sun"
    )

    print(
        f"Median and percentiles: "
        f"{star['R_IR_median']:.4f} "
        f"[{star['R_IR_p16']:.4f}, {star['R_IR_p84']:.4f}] R_sun"
    )

    print(f"Z_IR = {star['Z_IR']:.4f}")

    print(
        f"\nReference radius: "
        f"R_A = {star['R_A']:.4f} +- {star['e_R_A']:.4f} R_sun"
    )

    if abs(star["Z_IR"]) < 1.96:
        print("Final IR Z-test result: success")
    else:
        print("Final IR Z-test result: failed")


# ==========================================================
# PLOTTING
# ==========================================================
def plot_multiband_radius_for_star(data, index):
    """
    Plot deterministic radius estimates from all photometric bands.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    star = data.iloc[index]

    bands = list(BANDS.keys())
    x = np.arange(len(bands))

    y = np.array([star[f"R_{band}_det"] for band in bands])
    y_error = np.array([star[f"e_R_{band}_det"] for band in bands])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.errorbar(
        x,
        y,
        yerr=y_error,
        fmt="k.",
        label="Deterministic band estimates",
    )

    ax.scatter(
        x,
        y,
        c="gray",
        marker="o",
        edgecolors="k",
        s=45,
    )

    ax.axhline(
        star["R_A"],
        linestyle="--",
        label="Reference radius",
    )

    ax.axhline(
        star["R_IR"],
        linestyle="-",
        label="Final IR radius",
    )

    ax.set_title(f"Multi-band radius estimates: {star['ID']}")
    ax.set_xlabel("Photometric band")
    ax.set_ylabel(r"Radius [$R_\odot$]")
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend(fontsize=9)

    fig.tight_layout()

    output_file = OUTPUT_DIR / f"{star['ID']}_multiband_radius.pdf"
    fig.savefig(output_file, dpi=160)
    plt.show()

    return output_file


def plot_ir_radius_distribution(data, combined_samples, index):
    """
    Plot the final combined infrared Monte Carlo radius distribution.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    star = data.iloc[index]
    samples = combined_samples[index]

    if len(samples) == 0:
        print(f"No valid IR Monte Carlo samples for {star['ID']}.")
        return None

    n_bins = int(np.sqrt(len(samples)))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.hist(samples, bins=n_bins)

    ax.axvline(
        star["R_IR"],
        linestyle="-",
        label="Mean IR radius",
    )

    ax.axvline(
        star["R_A"],
        linestyle="--",
        label="Reference radius",
    )

    ax.set_title(f"Infrared radius distribution: {star['ID']}")
    ax.set_xlabel(r"Radius [$R_\odot$]")
    ax.set_ylabel("Counts")
    ax.legend(fontsize=9)

    fig.tight_layout()

    output_file = OUTPUT_DIR / f"{star['ID']}_IR_distribution.pdf"
    fig.savefig(output_file, dpi=160)
    plt.show()

    return output_file


def plot_final_radius_comparison(data):
    """
    Compare deterministic, infrared, and reference radius estimates.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.errorbar(
        data["R_A"],
        data["R_IR"],
        xerr=data["e_R_A"],
        yerr=data["e_R_IR"],
        fmt="k.",
        label="Stars",
    )

    ax.scatter(
        data["R_A"],
        data["R_IR"],
        c="gray",
        marker="o",
        edgecolors="k",
        s=38,
    )

    limit = 1.1 * max(
        np.nanmax(data["R_A"]),
        np.nanmax(data["R_IR"]),
    )

    reference = np.linspace(0.0, limit, 200)

    ax.plot(
        reference,
        reference,
        label="One-to-one relation",
    )

    ax.set_title("Final infrared radius comparison")
    ax.set_xlabel(r"Reference radius $R_A$ [$R_\odot$]")
    ax.set_ylabel(r"Infrared radius $R_{\rm IR}$ [$R_\odot$]")
    ax.legend(fontsize=9)

    fig.tight_layout()

    output_file = OUTPUT_DIR / "Final_R_IR_vs_R_A.pdf"
    fig.savefig(output_file, dpi=160)
    plt.show()

    return output_file


def plot_radius_residuals(data, threshold=1.96):
    """
    Plot residuals between reference and infrared radii.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x = data["R_IR"]
    y = data["R_A"] - data["R_IR"]
    y_error = np.sqrt(data["e_R_A"]**2 + data["e_R_IR"]**2)

    outlier_mask = np.abs(data["Z_IR"]) >= threshold

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.errorbar(
        x[~outlier_mask],
        y[~outlier_mask],
        yerr=y_error[~outlier_mask],
        fmt="k.",
        label="Accepted stars",
    )

    ax.scatter(
        x[~outlier_mask],
        y[~outlier_mask],
        c="gray",
        marker="o",
        edgecolors="k",
        s=38,
    )

    ax.errorbar(
        x[outlier_mask],
        y[outlier_mask],
        yerr=y_error[outlier_mask],
        fmt="r.",
        label="Failed Z-test",
    )

    ax.scatter(
        x[outlier_mask],
        y[outlier_mask],
        c="red",
        marker="o",
        edgecolors="r",
        s=38,
    )

    ax.axhline(0.0)

    ax.set_title("Infrared radius residuals")
    ax.set_xlabel(r"Infrared radius $R_{\rm IR}$ [$R_\odot$]")
    ax.set_ylabel(r"$R_A - R_{\rm IR}$ [$R_\odot$]")
    ax.legend(fontsize=9)

    fig.tight_layout()

    output_file = OUTPUT_DIR / "Residuals_R_A_minus_R_IR.pdf"
    fig.savefig(output_file, dpi=160)
    plt.show()

    return output_file


def plot_z_score_comparison(data):
    """
    Compare deterministic and infrared Z-scores.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.scatter(
        data["Z_all_det"],
        data["Z_IR"],
        c="gray",
        marker="o",
        edgecolors="k",
        s=38,
    )

    ax.axhline(0.0)
    ax.axvline(0.0)

    ax.axhline(1.96, linestyle="--")
    ax.axhline(-1.96, linestyle="--")
    ax.axvline(1.96, linestyle="--")
    ax.axvline(-1.96, linestyle="--")

    ax.set_title("Z-score comparison")
    ax.set_xlabel("Deterministic multi-band Z-score")
    ax.set_ylabel("Infrared Monte Carlo Z-score")

    fig.tight_layout()

    output_file = OUTPUT_DIR / "Z_score_comparison.pdf"
    fig.savefig(output_file, dpi=160)
    plt.show()

    return output_file


def plot_relative_radius_errors(data):
    """
    Compare relative uncertainties of the final infrared and reference radii.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x = 100.0 * data["e_R_IR"] / data["R_IR"]
    y = 100.0 * data["e_R_A"] / data["R_A"]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.scatter(
        x,
        y,
        c="gray",
        marker="o",
        edgecolors="k",
        s=38,
    )

    limit = 1.1 * max(np.nanmax(x), np.nanmax(y))
    reference = np.linspace(0.0, limit, 200)

    ax.plot(
        reference,
        reference,
        label="Equal relative uncertainty",
    )

    ax.set_title("Relative radius uncertainty comparison")
    ax.set_xlabel(r"Infrared radius relative uncertainty [%]")
    ax.set_ylabel(r"Reference radius relative uncertainty [%]")
    ax.legend(fontsize=9)

    fig.tight_layout()

    output_file = OUTPUT_DIR / "Relative_radius_errors.pdf"
    fig.savefig(output_file, dpi=160)
    plt.show()

    return output_file


def make_diagnostic_plots(data, combined_samples, index):
    """
    Generate all diagnostic plots.
    """

    print("\nGenerating diagnostic plots...")

    plot_files = []

    plot_files.append(plot_multiband_radius_for_star(data, index))
    plot_files.append(plot_ir_radius_distribution(data, combined_samples, index))
    plot_files.append(plot_final_radius_comparison(data))
    plot_files.append(plot_radius_residuals(data))
    plot_files.append(plot_z_score_comparison(data))
    plot_files.append(plot_relative_radius_errors(data))

    plot_files = [file for file in plot_files if file is not None]

    print("\nSaved plots:")

    for file in plot_files:
        print(f"- {file}")

    return plot_files


# ==========================================================
# ANALYSIS PIPELINE
# ==========================================================
def run_deterministic_analysis():
    """
    Run catalogue loading, validation, and deterministic multi-band analysis.
    """

    print("\nLoading stellar catalogue...")

    data = load_stellar_catalogue(DATA_FILE)

    print("\nValidating stellar catalogue...")

    data = validate_catalogue(data)

    print("\nComputing deterministic multi-band radii...")

    data = compute_all_band_radii(data)

    data = compute_weighted_multiband_radius(data)

    print("\nDeterministic multi-band analysis completed.")

    return data


def run_ir_monte_carlo_analysis(data):
    """
    Run infrared Monte Carlo radius estimation.
    """

    print("\nReading infrared bolometric corrections...")

    corrections = read_ir_bolometric_corrections(BC_FILE)

    bc_samples = reshape_ir_bolometric_corrections(
        corrections=corrections,
        data=data,
        n_samples=N_SAMPLES,
    )

    print("\nGenerating infrared Monte Carlo samples...")

    samples = sample_ir_observables(
        data=data,
        n_samples=N_SAMPLES,
        random_seed=RANDOM_SEED,
    )

    print("\nComputing infrared Monte Carlo radii...")

    (
        radius_samples,
        radius_mean,
        radius_std,
        radius_median,
        radius_p16,
        radius_p84,
    ) = compute_ir_radius_distributions(
        data=data,
        samples=samples,
        bc_samples=bc_samples,
    )

    (
        combined_samples,
        combined_mean,
        combined_std,
        combined_median,
        combined_p16,
        combined_p84,
    ) = combine_ir_radius_distributions(radius_samples)

    data = add_ir_results_to_catalogue(
        data=data,
        radius_mean=radius_mean,
        radius_std=radius_std,
        radius_median=radius_median,
        radius_p16=radius_p16,
        radius_p84=radius_p84,
        combined_mean=combined_mean,
        combined_std=combined_std,
        combined_median=combined_median,
        combined_p16=combined_p16,
        combined_p84=combined_p84,
    )

    print("\nInfrared Monte Carlo analysis completed.")

    return data, combined_samples


def run_full_analysis():
    """
    Run the full two-stage stellar radius analysis.

    Stage 1:
        Deterministic multi-band diagnostic estimate.

    Stage 2:
        Infrared Monte Carlo final estimate.
    """

    data = run_deterministic_analysis()

    data, combined_samples = run_ir_monte_carlo_analysis(data)

    save_results_table(
        data=data,
        output_file=RESULTS_FILE,
    )

    save_outlier_table(
        data=data,
        output_file=OUTLIER_FILE,
        threshold=1.96,
    )

    return data, combined_samples


# ==========================================================
# MAIN
# ==========================================================
def main():
    """
    Main execution function.
    """

    data, combined_samples = run_full_analysis()

    print_global_summary(data)

    print_failed_z_tests(
        data=data,
        column="Z_IR",
        threshold=1.96,
    )

    index = int(input("\nInsert star index for detailed diagnostics: "))

    if index < 0 or index >= len(data):
        raise ValueError(
            f"Invalid index. Choose a value between 0 and {len(data) - 1}."
        )

    print_star_summary(data, index)

    make_diagnostic_plots(
        data=data,
        combined_samples=combined_samples,
        index=index,
    )


if __name__ == "__main__":
    main()
