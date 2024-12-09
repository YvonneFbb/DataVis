#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click

def get_json_files_from_folder(folder_path):
    """
    Get all JSON file paths from a folder.
    :param folder_path: Path to the folder
    :return: List of all JSON file paths
    """
    return [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.endswith(".json")
    ]

def get_config(config_path):
    """
    Get the configuration from a JSON file.
    :param config_path: Path to the configuration file
    :return: Dictionary containing configuration data
    """
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config

def analyze_json_file(json_file_path, analysis_type):
    """
    Analyze a JSON file to find the maximum width and calculate the distribution.
    :param json_file_path: Path to the JSON file
    :param analysis_type: Type of analysis ('literal', 'aspect', or 'dark')
    :return: Calculated distribution for the given JSON file
    """
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    if analysis_type == "literal":
        widths = [
            item["width"] for item in data if "width" in item and "height" in item
        ]
        max_width = max(widths) if widths else 0

        if max_width == 0:
            return []

        distributions = [
            (item["width"] * item["height"]) / (max_width**2)
            for item in data
            if "width" in item and "height" in item
        ]
    elif analysis_type == "aspect":
        distributions = [
            item["height"] / item["width"] if item["width"] != 0 else 0
            for item in data
            if "width" in item and "height" in item
        ]
    elif analysis_type == "dark":
        widths = [item["width"] for item in data if "width" in item]
        max_width = max(widths) if widths else 0

        if max_width == 0:
            return []

        distributions = [
            item["dark_pixels"] / (max_width**2)
            for item in data
            if "dark_pixels" in item and "width" in item
        ]
    else:
        raise ValueError("Invalid analysis type. Use 'literal', 'aspect', or 'dark'.")

    return distributions

def analyze_all_json_files(folder_path, analysis_type, config):
    """
    Analyze all JSON files in a folder according to a specified configuration.
    :param folder_path: Path to the folder containing JSON files
    :param analysis_type: Type of analysis ('literal', 'aspect', or 'dark')
    :param config: Configuration dictionary specifying file order and colors
    :return: A dictionary with file names as keys and distributions as values
    """
    json_files = get_json_files_from_folder(folder_path)
    json_files.sort()  # Sort JSON files by file name
    ordered_files = config.get("order", json_files)
    analysis_results = {}

    for json_file in ordered_files:
        json_file_path = os.path.join(folder_path, json_file)
        if json_file_path in json_files:
            try:
                distributions = analyze_json_file(json_file_path, analysis_type)
                if distributions:
                    analysis_results[json_file] = distributions
            except (ValueError, KeyError) as e:
                print(f"Error processing file {json_file}: {e}")

    return analysis_results

def plot_distributions(analysis_results, result_folder, analysis_type):
    """
    Plot the distributions for each JSON file and save them to the result folder.
    :param analysis_results: Dictionary with file names as keys and distributions as values
    :param result_folder: Path to the folder where results will be saved
    :param analysis_type: Type of analysis ('literal', 'aspect', or 'dark')
    """
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for json_file, distributions in analysis_results.items():
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(
            distributions, bins=30, alpha=0.75, edgecolor="black"
        )
        xlabel = (
            "Width * Height / Max Width^2"
            if analysis_type == "literal"
            else (
                "Height / Width"
                if analysis_type == "aspect"
                else "Dark Pixels / Max Width^2"
            )
        )
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.title(f"Distribution for {os.path.basename(json_file)}")
        plt.grid(True)

        # Annotate bars with ratio value if frequency is greater than 0
        for i in range(len(patches)):
            if patches[i].get_height() > 0:
                bin_center = (bins[i] + bins[i + 1]) / 2
                plt.text(
                    bin_center,
                    patches[i].get_height(),
                    f"{bin_center:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        output_path = os.path.join(
            result_folder,
            f"{os.path.basename(json_file).replace('.json', '')}_distribution_{analysis_type}.png",
        )
        plt.savefig(output_path, dpi=300)
        plt.close()


def plot_combined_statistics(analysis_results, result_folder, analysis_type, config):
    """
    Plot combined comparisons for mean, median, IQR, variance, and maximum/minimum values across all JSON files.
    :param analysis_results: Dictionary with file names as keys and distributions as values
    :param result_folder: Path to the folder where the result will be saved
    :param analysis_type: Type of analysis ('literal', 'aspect', or 'dark')
    :param config: Configuration dictionary specifying colors
    """
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    file_names = [os.path.basename(json_file) for json_file in analysis_results.keys()]
    x = np.arange(len(file_names))
    width = 0.08  # Make the box plot narrower for a more compact layout and reduce spacing

    plt.figure(figsize=(15, 10))

    # Increase spacing between box plots by adjusting the positions
    spacing = 0.3
    positions = x * spacing

    # Set the color palette for the box plot
    color_groups = [[(179 / 255, 219 / 255, 255 / 255), (0 / 255, 91 / 255, 191 / 255)],
                    [(255 / 255, 198 / 255, 194 / 255), (183 / 255, 24 / 255, 11 / 255)],
                    [(255 / 255, 221 / 255, 176 / 255), (211 / 255, 111 / 255, 2 / 255)],
                    [(219 / 255, 219 / 255, 219 / 255), (146 / 255, 146 / 255, 146 / 255)]]

    # Set the color palette for the box plot
    box_colors = [color_groups[config.get("colors", {}).get(file_name, 0)][0] for file_name in file_names]
    dot_colors = [color_groups[config.get("colors", {}).get(file_name, 0)][1] for file_name in file_names]

    # Median with IQR using Box Plot representation
    boxplot = plt.boxplot(
        [analysis_results[json_file] for json_file in analysis_results.keys()],
        positions=positions,
        widths=width,
        patch_artist=True,
        showmeans=True,
        medianprops={"color": "red"},
        meanprops={"marker": "o", "markerfacecolor": "blue", "markersize": 5},
        whiskerprops={"linestyle": "-"},
        capprops={"linestyle": "-"},
    )

    # Set colors for each box
    for patch, color in zip(boxplot['boxes'], box_colors):
        patch.set_facecolor(color)

    # Annotate median values
    for i, line in enumerate(boxplot["medians"]):
        median_value = line.get_ydata()[0]
        plt.text(
            positions[i] + width,
            median_value,
            f"{median_value:.2f}",
            verticalalignment="bottom",
            color="red",
        )

    ylabel = (
        "Value Comparison (literal)"
        if analysis_type == "literal"
        else (
            "Value Comparison (Aspect Ratio)"
            if analysis_type == "aspect"
            else "Value Comparison (Dark Pixels)"
        )
    )
    plt.xlabel("JSON Files")
    plt.ylabel(ylabel)
    plt.title("Combined Statistical Comparisons across JSON Files")
    plt.xticks(positions, file_names, rotation=90, ha="right")
    plt.tight_layout()

    # Remove grid lines for a cleaner look
    plt.grid(False)

    # Plot all data points in black color
    for i, json_file in enumerate(analysis_results.keys()):
        distributions = analysis_results[json_file]
        plt.scatter(
            [positions[i]] * len(distributions),
            distributions,
            alpha=0.2,
            s=10,
            color=dot_colors[i],
        )

    output_path = os.path.join(
        result_folder, f"combined_statistical_comparison_{analysis_type}.png"
    )
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_distribution_overlap(analysis_results, result_folder, analysis_type):
    """
    Plot the overlap of distributions across all JSON files using KDE plots.
    :param analysis_results: Dictionary with file names as keys and distributions as values
    :param result_folder: Path to the folder where the result will be saved
    :param analysis_type: Type of analysis ('literal', 'aspect', or 'dark')
    """
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    plt.figure(figsize=(15, 10))
    for json_file, distributions in analysis_results.items():
        if distributions:
            sns.kdeplot(
                distributions, label=json_file, fill=True, alpha=0.3
            )

    xlabel = (
        "Width * Height / Max Width^2"
        if analysis_type == "literal"
        else (
            "Height / Width"
            if analysis_type == "aspect"
            else "Dark Pixels / Max Width^2"
        )
    )
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title("Distribution Overlap across JSON Files")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(
        result_folder, f"distribution_overlap_{analysis_type}.png"
    )
    plt.savefig(output_path, dpi=300)
    plt.close()

@click.command()
@click.argument("folder_path", type=click.Path(exists=True))
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--analysis-type",
    type=click.Choice(["literal", "aspect", "dark"]),
    help="Type of analysis to perform: literal, aspect, or dark",
)
def main(folder_path, config_path, analysis_type):
    config = get_config(config_path)
    result_folder = os.path.join(folder_path, f"{analysis_type}_results")
    analysis_results = analyze_all_json_files(folder_path, analysis_type, config)

    if analysis_results:
        plot_distributions(analysis_results, result_folder, analysis_type)
        plot_combined_statistics(analysis_results, result_folder, analysis_type, config)
        plot_distribution_overlap(analysis_results, result_folder, analysis_type)
        print(f"Results saved to {result_folder}")
    else:
        print("No valid JSON files found or unable to process files.")

if __name__ == "__main__":
    main()
