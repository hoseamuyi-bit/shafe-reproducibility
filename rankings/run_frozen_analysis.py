"""Run the frozen analysis with tuple tolerances accepted by NumPy."""
from unittest.mock import patch

import numpy as np

import analyze_ranking


def run():
    frozen_sign = analyze_ranking.sign

    def sign_with_array_tolerance(delta, tolerance):
        return frozen_sign(delta, np.asarray(tolerance))

    with patch.object(analyze_ranking, 'sign', sign_with_array_tolerance):
        analyze_ranking.analyze()


if __name__ == '__main__':
    run()
