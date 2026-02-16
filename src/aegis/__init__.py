"""AEGIS package API."""

from .bounds import (
    negative_control_sensitivity_bound,
    policy_regret_minimax_floor,
    policy_regret_upper_bound,
)
from .monitor import (
    AEGISMonitor,
    deficiency_proxy_propensity,
    fit_logistic_newton,
    negative_control_distance,
    negative_control_tv,
    predict_propensity,
    uniform_deficiency_proxy,
)
from .theory import (
    confounding_frontier,
    overlap_diagnostic,
    partial_id_set,
    policy_regret_vc_bound,
    rkhs_rate_bound,
    sharp_lower_bound,
    wasserstein_deficiency_gaussian,
)
from .version import __version__

__all__ = [
    "__version__",
    "AEGISMonitor",
    "deficiency_proxy_propensity",
    "fit_logistic_newton",
    "negative_control_distance",
    "negative_control_tv",
    "predict_propensity",
    "uniform_deficiency_proxy",
    "negative_control_sensitivity_bound",
    "policy_regret_upper_bound",
    "policy_regret_minimax_floor",
    "rkhs_rate_bound",
    "policy_regret_vc_bound",
    "wasserstein_deficiency_gaussian",
    "sharp_lower_bound",
    "partial_id_set",
    "overlap_diagnostic",
    "confounding_frontier",
]
