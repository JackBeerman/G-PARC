"""
visualizations.selection
========================
Select representative, best, or worst simulations for visualization.
"""

__all__ = ['select_representative_simulations']


def select_representative_simulations(sim_metrics_list, n_samples=3,
                                       selection_mode='representative',
                                       sort_key='rmse'):
    """
    Select simulation indices for visualization.

    Args:
        sim_metrics_list: list of dicts; each must have 'overall' with sort_key,
                          OR directly have sort_key at top level (for flat dicts).
        n_samples: how many to select
        selection_mode: 'representative' (best+median+worst), 'best', 'worst', 'all'
        sort_key: key to sort by (ascending = best first)

    Returns:
        list of indices into sim_metrics_list
    """
    if not sim_metrics_list:
        return []

    # Support both nested {'overall': {'rmse': ...}} and flat {'mean_rmse': ...}
    def _get_val(m):
        if 'overall' in m and isinstance(m['overall'], dict):
            return m['overall'].get(sort_key, m['overall'].get('rmse', 0))
        return m.get(sort_key, m.get('mean_rmse', 0))

    sims = sorted(enumerate(sim_metrics_list), key=lambda x: _get_val(x[1]))

    if selection_mode == 'all':
        return [s[0] for s in sims]
    elif selection_mode == 'best':
        return [s[0] for s in sims[:n_samples]]
    elif selection_mode == 'worst':
        return [s[0] for s in sims[-n_samples:]]
    else:  # representative
        selected = []
        if len(sims) >= 1:
            selected.append(sims[0][0])
        if len(sims) >= 2:
            selected.append(sims[len(sims) // 2][0])
        if len(sims) >= 3:
            selected.append(sims[-1][0])
        return selected[:n_samples]
