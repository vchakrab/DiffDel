#!/usr/bin/env python3
"""
constraint_cells_simple.py
==========================

Simple script that just prints constraint cells using the existing logic.
"""

import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from cell import Cell, Attribute
from rtf_core.config import get_dataset_info
from importlib import import_module


def discover_constraint_cells(target_cell_info, dataset='adult'):
    """Use the existing _discover_constraint_cells logic."""
    
    # Get dataset info
    dataset_info = get_dataset_info(dataset)
    table_name = dataset_info['primary_table']
    
    # Load denial constraints
    dc_module_path = dataset_info.get('dc_config_module')
    denial_constraints = []
    if dc_module_path:
        dc_module = import_module(dc_module_path)
        denial_constraints = getattr(dc_module, 'denial_constraints', [])
    
    # Mock row data
    mock_row_data = {
        'age': 25, 'workclass': 'Private', 'education': 'Bachelors', 'education_num': 13,
        'marital_status': 'Never-married', 'occupation': 'Tech-support', 'relationship': 'Not-in-family',
        'race': 'White', 'sex': 'Male', 'capital_gain': 0, 'capital_loss': 0,
        'hours_per_week': 40, 'native_country': 'United-States', 'income': '<=50K'
    }
    
    # Create target cell
    target_cell = Cell(
        Attribute(table_name, target_cell_info['attribute']),
        target_cell_info['key'],
        mock_row_data[target_cell_info['attribute']]
    )
    
    # This is the exact logic from _discover_constraint_cells
    target_attr = target_cell.attribute.col
    related_attrs = set()
    
    for dc in denial_constraints:
        attrs_in_dc = set(pred.split('.')[-1] for pred in [p[0] for p in dc] + [p[2] for p in dc if isinstance(p[2], str)])
        if target_attr in attrs_in_dc:
            related_attrs.update(attrs_in_dc)
    
    related_attrs.discard(target_attr)
    
    constraint_cells = set()
    for attr in related_attrs:
        if attr in mock_row_data:
            constraint_cell = Cell(Attribute(table_name, attr), target_cell_info['key'], mock_row_data[attr])
            constraint_cells.add(constraint_cell)
    
    return constraint_cells


def print_constraint_cells(target_cell_info, dataset='adult'):
    """Print constraint cells."""
    
    print(f"Target: {target_cell_info}")
    
    constraint_cells = discover_constraint_cells(target_cell_info, dataset)
    
    print(f"Constraint cells: {len(constraint_cells)}")
    for cell in constraint_cells:
        print(f"  {cell.attribute.col} = {cell.value}")


if __name__ == '__main__':
    # Test different targets
    targets = [
        {'key': 2, 'attribute': 'education'},
        {'key': 5, 'attribute': 'age'},
        {'key': 10, 'attribute': 'occupation'}
    ]
    
    for target in targets:
        print_constraint_cells(target)
        print()


