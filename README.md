# PSI class

import numpy as np
import pandas as pd

class PSICalculator:
    """
    A class for calculating the Population Stability Index (PSI) for model attributes.
    
    Attributes:
        bins (int): The default number of bins to use for PSI calculation.
    """

    def __init__(self, bins=10):
        """
        Initializes the PSICalculator with a default number of bins.
        
        :param bins: Default number of bins for PSI calculation.
        """
        self.bins = bins

    def calculate_psi(self, expected, actual, bins=None):
        """
        Calculate the PSI for a single attribute.

        :param expected: Array of expected values
        :param actual: Array of actual values
        :param bins: Number of bins to use, defaults to the class attribute
        :return: PSI value
        """
        if bins is None:
            bins = self.bins

        expected_counts, bin_edges = np.histogram(expected, bins=bins, range=(0, 1))
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        psi_values = []
        for e_count, a_count in zip(expected_counts, actual_counts):
            e_pct = e_count / len(expected)
            a_pct = a_count / len(actual)

            if e_pct == 0:
                psi_value = 0
            else:
                psi_value = (e_pct - a_pct) * np.log(e_pct / a_pct)

            psi_values.append(psi_value)

        psi_total = sum(psi_values)
        return psi_total

    def calculate_psi_multiple_attributes(self, expected_df, actual_df, attributes):
        """
        Calculate PSI for multiple attributes.

        :param expected_df: DataFrame with expected values
        :param actual_df: DataFrame with actual values
        :param attributes: List of attribute names
        :return: Dictionary with PSI values for each attribute
        """
        psi_values = {}
        for attribute in attributes:
            expected = expected_df[attribute].values
            actual = actual_df[attribute].values

            psi_value = self.calculate_psi(expected, actual)
            psi_values[attribute] = psi_value

        return psi_values

# Example usage (commented out)
# psi_calculator = PSICalculator(bins=10)
# attributes_to_calculate = ['attribute1', 'attribute2', 'attribute3']
# psi_values = psi_calculator.calculate_psi_multiple_attributes(expected_dataset, actual_dataset, attributes_to_calculate)
# print(psi_values)
