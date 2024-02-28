#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:38:04 2024

@author: albertovth
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.linalg import eig
import matplotlib.pyplot as plt

def calculate_priority_vector(matrix):
    try:
        eigvals, eigvecs = eig(matrix)
        max_eigval = np.max(eigvals.real)
        max_eigvec = eigvecs[:, eigvals.real.argmax()].real
        priority_vector = max_eigvec / np.sum(max_eigvec)
        return priority_vector
    except Exception as e:
        st.error(f"Error in calculating priority vector: {e}")
        return np.array([])

def create_pairwise_comparison_matrix_from_grades(elements, grades):
    n = len(elements)
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            grade = grades.get((elements[i], elements[j]), 1)  # Default to 1 if not specified
            matrix[i, j] = grade
            matrix[j, i] = 1 / grade if grade != 0 else 0
    return matrix

def input_to_grades_dict(input_text, elements):
    grades = {}
    for line in input_text.split('\n'):
        parts = line.split(',')
        if len(parts) == 3:
            elem1, elem2, grade_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
            try:
                grade = float(grade_str)
            except ValueError:
                st.warning(f"Invalid grade value for {elem1}, {elem2}. Using default value 1.")
                grade = 1
            if elem1 in elements and elem2 in elements:  # Validates elements
                grades[(elem1, elem2)] = grade
    return grades

def app():
    st.title('AHP Analysis with Criteria-Alternative Relationships')

    criteria_input = st.text_input('Enter criteria separated by comma', 'Criterion A,Criterion B').split(',')
    criteria_input = [c.strip() for c in criteria_input]
    alternatives_input = st.text_input('Enter alternatives separated by comma', 'Alternative X,Alternative Y,Alternative Z').split(',')
    alternatives_input = [a.strip() for a in alternatives_input]

    criteria_alternative_relationship = {}
    for criterion in criteria_input:
        alternatives_for_criterion = st.text_input(f'Enter alternatives related to {criterion} separated by comma', 'Alternative X,Alternative Y').split(',')
        criteria_alternative_relationship[criterion] = [a.strip() for a in alternatives_for_criterion]

    criteria_achievability_grades_input = st.text_area('Enter grades for pairwise comparisons of criteria on Achievability, separated by comma')
    criteria_effect_grades_input = st.text_area('Enter grades for pairwise comparisons of criteria on Effect, separated by comma')

    criteria_achievability_grades = input_to_grades_dict(criteria_achievability_grades_input, criteria_input)
    criteria_effect_grades = input_to_grades_dict(criteria_effect_grades_input, criteria_input)

    alternative_grades = {criterion: {'Achievability': {}, 'Effect': {}} for criterion in criteria_input}

    for criterion, related_alternatives in criteria_alternative_relationship.items():
        achievability_input = st.text_area(f'Enter grades for pairwise comparisons of alternatives on Achievability for {criterion}, separated by comma')
        effect_input = st.text_area(f'Enter grades for pairwise comparisons of alternatives on Effect for {criterion}, separated by comma')
        alternative_grades[criterion]['Achievability'] = input_to_grades_dict(achievability_input, related_alternatives)
        alternative_grades[criterion]['Effect'] = input_to_grades_dict(effect_input, related_alternatives)
    # Continuing from the last part of provided code
    if st.button('Calculate Weights'):
        try:
            criteria_achievability_matrix = create_pairwise_comparison_matrix_from_grades(criteria_input, criteria_achievability_grades)
            criteria_effect_matrix = create_pairwise_comparison_matrix_from_grades(criteria_input, criteria_effect_grades)   
            criteria_achievability_vector = calculate_priority_vector(criteria_achievability_matrix) * 100 / sum(calculate_priority_vector(criteria_achievability_matrix))
            criteria_effect_vector = calculate_priority_vector(criteria_effect_matrix) * 100 / sum(calculate_priority_vector(criteria_effect_matrix))
            
            synthetic_alternative_achievability_weights = {}
            synthetic_alternative_effect_weights = {}
            
            for criterion, alternatives in criteria_alternative_relationship.items():
                achievability_matrix = create_pairwise_comparison_matrix_from_grades(alternatives, alternative_grades[criterion]['Achievability'])
                effect_matrix = create_pairwise_comparison_matrix_from_grades(alternatives, alternative_grades[criterion]['Effect'])
                
                alternative_achievability_vector = calculate_priority_vector(achievability_matrix)
                alternative_effect_vector = calculate_priority_vector(effect_matrix)
                
                index = criteria_input.index(criterion)
                criteria_weight_for_achievability = criteria_achievability_vector[index]
                criteria_weight_for_effect = criteria_effect_vector[index]
                
                for i, alternative in enumerate(alternatives):
                    synthetic_alternative_achievability_weights[alternative] = synthetic_alternative_achievability_weights.get(alternative, 0) + alternative_achievability_vector[i] * criteria_weight_for_achievability
                    synthetic_alternative_effect_weights[alternative] = synthetic_alternative_effect_weights.get(alternative, 0) + alternative_effect_vector[i] * criteria_weight_for_effect
            
            # Consolidating weights
            consolidated_achievability_weights = pd.Series(synthetic_alternative_achievability_weights).sort_values(ascending=False)
            consolidated_effect_weights = pd.Series(synthetic_alternative_effect_weights).sort_values(ascending=False)
            
            # Display tables
            st.table(consolidated_achievability_weights.reset_index().rename(columns={'index': 'Alternative', 0: 'Synthetic Achievability Weight'}))
            st.table(consolidated_effect_weights.reset_index().rename(columns={'index': 'Alternative', 0: 'Synthetic Effect Weight'}))
        
            # Creating a figure and a grid of subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))  # Adjusted for a bigger figure

            
            # Main scatter plot on ax1
            median_achievability = consolidated_achievability_weights.median()
            median_effect = consolidated_effect_weights.median()
            ax1.axvline(x=median_achievability, color='red', linestyle='--', label='Median Achievability')
            ax1.axhline(y=median_effect, color='red', linestyle='--', label='Median Effect')
            
            # Fill quadrants with specific colors
            # Top Priority quadrant
            ax1.fill_between([median_achievability, max(consolidated_achievability_weights.values)], median_effect, max(consolidated_effect_weights.values), color='#78C850', alpha=0.3)
            # Priority quadrants
            ax1.fill_between([0, median_achievability], median_effect, max(consolidated_effect_weights.values), color='#FDB147', alpha=0.3)
            ax1.fill_between([median_achievability, max(consolidated_achievability_weights.values)], 0, median_effect, color='#FDB147', alpha=0.3)
            # Not Priority quadrant
            ax1.fill_between([0, median_achievability], 0, median_effect, color='#E57373', alpha=0.3)
            
            # Plotting points and annotations
            for alternative, achievability_weight in zip(consolidated_achievability_weights.index, consolidated_achievability_weights):
                effect_weight = consolidated_effect_weights[alternative]
                ax1.scatter(achievability_weight, effect_weight, color='blue')
                ax1.annotate(alternative, (achievability_weight, effect_weight), textcoords="offset points", xytext=(10,10), ha='right')
            
            ax1.set_xlabel('Synthetic Alternative Achievability Weights')
            ax1.set_ylabel('Synthetic Alternative Effect Weights')
            ax1.set_title('Scatter Plot of Synthetic Weights')
            ax1.legend()
            
            # Using ax2 for informational panel or legend with background colors
            ax2.axis('off')  # Turn off axis
            
            # Add texts with background colors
            bgcolor = ['#78C850', '#FDB147', '#E57373']
            text = ['Top Priority - Light Green', 'Priority - Light Yellow', 'Not Priority - Light Red']
            y_positions = [0.6, 0.5, 0.4]
            
            for bg, txt, y in zip(bgcolor, text, y_positions):
                ax2.text(0.5, y, txt, horizontalalignment='center', verticalalignment='center', fontsize=10, bbox=dict(facecolor=bg, alpha=0.5, edgecolor='none'))
            
            plt.tight_layout()
            st.pyplot(fig)



        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    app()