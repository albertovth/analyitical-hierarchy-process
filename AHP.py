#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:36:47 2024

@author: albertovth
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:36:47 2024

@author: albertovth
"""
import streamlit as st
import numpy as np
import pandas as pd
from scipy.linalg import eig
import matplotlib.pyplot as plt
import pandas as pd

def calculate_priority_vector(matrix):
    eigvals, eigvecs = eig(matrix)
    max_eigval = np.max(eigvals.real)
    max_eigvec = eigvecs[:, eigvals.real.argmax()].real
    priority_vector = max_eigvec / np.sum(max_eigvec)
    return priority_vector

def create_pairwise_comparison_matrix_from_grades(elements, grades):
    n = len(elements)
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            grade = grades.get((elements[i], elements[j]), 1)  # Default to 1 if not specified
            matrix[i, j] = grade
            matrix[j, i] = 1 / grade
    return matrix

def input_to_grades_dict(input_text, elements):
    grades = {}
    for line in input_text.split('\n'):
        parts = line.split(',')
        if len(parts) == 3:
            elem1, elem2, grade = parts[0].strip(), parts[1].strip(), float(parts[2].strip())
            if elem1 in elements and elem2 in elements:  # Validates elements
                grades[(elem1, elem2)] = grade
    return grades

def app():
    st.title('AHP Analysis with Criteria-Alternative Relationships')

    # Input for Criteria and Alternatives
    criteria_input = st.text_input('Enter criteria separated by comma', 'Criterion A,Criterion B').split(',')
    criteria_input = [c.strip() for c in criteria_input]
    alternatives_input = st.text_input('Enter alternatives separated by comma', 'Alternative X,Alternative Y,Alternative Z').split(',')
    alternatives_input = [a.strip() for a in alternatives_input]

    # Establish relationships between criteria and alternatives
    criteria_alternative_relationship = {}
    for criterion in criteria_input:
        alternatives_for_criterion = st.text_input(f'Enter alternatives related to {criterion} separated by comma', 'Alternative X,Alternative Y').split(',')
        criteria_alternative_relationship[criterion] = [a.strip() for a in alternatives_for_criterion]

    # Input for Grades for Pairwise Comparisons
    criteria_achievability_grades_input = st.text_area('Enter grades for pairwise comparisons of criteria on Achievability, separated by comma')
    criteria_effect_grades_input = st.text_area('Enter grades for pairwise comparisons of criteria on Effect, separated by comma')

    # Convert criteria input text to dictionaries
    criteria_achievability_grades = input_to_grades_dict(criteria_achievability_grades_input, criteria_input)
    criteria_effect_grades = input_to_grades_dict(criteria_effect_grades_input, criteria_input)

    # Placeholder for alternative grades input (to be filled based on criteria-alternative relationship)
    alternative_grades = {criterion: {'Achievability': {}, 'Effect': {}} for criterion in criteria_input}

    for criterion, related_alternatives in criteria_alternative_relationship.items():
        achievability_input = st.text_area(f'Enter grades for pairwise comparisons of alternatives on Achievability for {criterion}, separated by comma')
        effect_input = st.text_area(f'Enter grades for pairwise comparisons of alternatives on Effect for {criterion}, separated by comma')
        alternative_grades[criterion]['Achievability'] = input_to_grades_dict(achievability_input, related_alternatives)
        alternative_grades[criterion]['Effect'] = input_to_grades_dict(effect_input, related_alternatives)

    if st.button('Calculate Weights'):
        # Calculate Criteria Weights
        criteria_achievability_matrix = create_pairwise_comparison_matrix_from_grades(criteria_input, criteria_achievability_grades)
        criteria_effect_matrix = create_pairwise_comparison_matrix_from_grades(criteria_input, criteria_effect_grades)
        criteria_achievability_vector = calculate_priority_vector(criteria_achievability_matrix) * 100 / sum(calculate_priority_vector(criteria_achievability_matrix))
        criteria_effect_vector = calculate_priority_vector(criteria_effect_matrix) * 100 / sum(calculate_priority_vector(criteria_effect_matrix))
        
        
        # Initialize dictionaries to hold final synthetic weights
        final_synthetic_weights_achievability = {}
        final_synthetic_weights_effect = {}

        # Loop over each criterion to calculate synthetic weights for connected alternatives
        for criterion, alternatives in criteria_alternatives_map.items():
            criterion_weight_achievability = criteria_weights_achievability[criterion]
            criterion_weight_effect = criteria_weights_effect[criterion]

        for alternative in alternatives:
            alternative_weight_achievability = alternative_weights_achievability[criterion][alternative]
            alternative_weight_effect = alternative_weights_effect[criterion][alternative]

            # Aggregate synthetic weights for alternatives appearing under multiple criteria
            if alternative in final_synthetic_weights_achievability:
                final_synthetic_weights_achievability[alternative] += alternative_weight_achievability * criterion_weight_achievability
            else:
                final_synthetic_weights_achievability[alternative] = alternative_weight_achievability * criterion_weight_achievability

            if alternative in final_synthetic_weights_effect:
                final_synthetic_weights_effect[alternative] += alternative_weight_effect * criterion_weight_effect
            else:
                final_synthetic_weights_effect[alternative] = alternative_weight_effect * criterion_weight_effect

        # Normalize the synthetic weights
        total_achievability = sum(final_synthetic_weights_achievability.values())
        total_effect = sum(final_synthetic_weights_effect.values())

        normalized_synthetic_weights_achievability = {k: v / total_achievability * 100 for k, v in final_synthetic_weights_achievability.items()}
        normalized_synthetic_weights_effect = {k: v / total_effect * 100 for k, v in final_synthetic_weights_effect.items()}
        
        # Convert to DataFrame for display
        df_achievability = pd.DataFrame(list(normalized_synthetic_weights_achievability.items()), columns=['Alternative', 'Achievability'])
        df_effect = pd.DataFrame(list(normalized_synthetic_weights_effect.items()), columns=['Alternative', 'Effect'])

        # Sort and display
        df_achievability.sort_values(by='Achievability', ascending=False, inplace=True)
        df_effect.sort_values(by='Effect', ascending=False, inplace=True)

        st.write("Achievability:")
        st.dataframe(df_achievability)
        st.write("\nEffect:")
        st.dataframe(df_effect)
           
        # Recalculate synthetic weights to account for alternatives appearing under multiple criteria
        adjusted_synthetic_weights_achievability = {}
        adjusted_synthetic_weights_effect = {}

        for alternative in synthetic_weights_achievability.keys():
            # Assuming alternative_grades contains the grades for alternatives under each criterion
            criteria_count = sum(alternative in grades for criterion, grades in alternative_grades.items() if 'Achievability' in grades)
            if criteria_count > 0:  # If the alternative appears in more than one criterion
                adjusted_synthetic_weights_achievability[alternative] = synthetic_weights_achievability[alternative]
            else:  # If the alternative is unique to a single criterion
                adjusted_synthetic_weights_achievability[alternative] = synthetic_weights_achievability[alternative]

        for alternative in synthetic_weights_effect.keys():
            criteria_count = sum(alternative in grades for criterion, grades in alternative_grades.items() if 'Effect' in grades)
            if criteria_count > 0:  # If the alternative appears in more than one criterion
                adjusted_synthetic_weights_effect[alternative] = synthetic_weights_effect[alternative]
            else:  # If the alternative is unique to a single criterion
                adjusted_synthetic_weights_effect[alternative] = synthetic_weights_effect[alternative]

        # Calculate new totals for normalization
        total_achievability = sum(adjusted_synthetic_weights_achievability.values())
        total_effect = sum(adjusted_synthetic_weights_effect.values())

        # Normalize synthetic weights
        synthetic_weights_achievability = {k: v / total_achievability * 100 for k, v in adjusted_synthetic_weights_achievability.items()}
        synthetic_weights_effect = {k: v / total_effect * 100 for k, v in adjusted_synthetic_weights_effect.items()}

        # Display results in table
        df_results = pd.DataFrame(list(zip(synthetic_weights_achievability.keys(), synthetic_weights_achievability.values(), synthetic_weights_effect.values())), columns=['Alternative', 'Achievability (%)', 'Effect (%)'])
        st.write('Alternative Synthetic Weights', df_results)
        
        # Scatter Plot
        plt.scatter(df_achievability['Achievability'], df_effect['Effect'])
        plt.axvline(x=df_achievability['Achievability'].median(), color='r', linestyle='--')
        plt.axhline(y=df_effect['Effect'].median(), color='r', linestyle='--')
        plt.xlabel('Achievability (%)')
        plt.ylabel('Effect (%)')
        plt.title('Scatter Plot of Alternatives')
        st.pyplot(plt)

if __name__ == '__main__':
    app()