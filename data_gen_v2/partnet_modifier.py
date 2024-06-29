"""Modifier for better visualization of PartNet dataset."""


def limit_modifier(model_cat, joint_type, cur_limit):
    """Modify the limit for a specific joint."""
    if model_cat == "KitchenPot":
        if joint_type == "prismatic":
            return [cur_limit[0], cur_limit[1] * 10]
    return cur_limit
