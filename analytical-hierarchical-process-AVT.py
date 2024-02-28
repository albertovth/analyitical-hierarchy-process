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

import numpy as np

def create_pairwise_comparison_matrix_from_grades(elements, grades):
    n = len(elements)
    matrix = np.ones((n, n))  
    for i in range(n):
        for j in range(i+1, n):  
            grade_i = grades.get(elements[i], 1)  
            grade_j = grades.get(elements[j], 1)  
            
            if grade_j != 0:  
                matrix[i, j] = grade_i / grade_j
                matrix[j, i] = grade_j / grade_i
            else:
                matrix[i, j] = 0  
                matrix[j, i] = 0 if grade_i == 0 else np.inf  

    return matrix

def input_to_grades_dict(input_text, elements):
    grades = {}
    
    try:
        grades_list = [float(grade.strip()) for grade in input_text.split(',')]
    except ValueError as e:
        st.error(f"Invalid grade value encountered: {e}")
        return grades  

    
    if len(grades_list) != len(elements):
        st.error("The number of grades does not match the number of elements")
        return grades  

    
    for i, grade in enumerate(grades_list):
        grades[elements[i]] = grade

    return grades

def input_to_grades_dict_for_alternatives(input_text, alternatives):
   grades = {}
   
   try:
       grades_list = [float(grade.strip()) for grade in input_text.split(',')]
   except ValueError as e:
       st.error(f"Invalid grade value encountered: {e}")
       return grades  

   
   if len(grades_list) != len(alternatives):
       st.error("The number of grades does not match the number of elements")
       return grades  

   
   for i, grade in enumerate(grades_list):
       grades[alternatives[i]] = grade

   return grades


def app():
    
    st.title('Analytical Hierarchical Network with Criteria-Alternative Relationships')

    criteria_achievability_input = st.text_input('Enter criteria separated by comma', 'Higher revenue, Higher profit, Minimal environmental footprint, Social responsibility').split(',')
    criteria_achievability_input = [c.strip() for c in criteria_achievability_input]
    criteria_effect_input = criteria_achievability_input.copy() 
    criteria_effect_input = [c.strip() for c in criteria_effect_input]
     
    alternatives_achievability_input = st.text_input('Enter alternatives separated by comma', 'Project 1, Project 2, Project 3, Project 4, Project 5').split(',')
    alternatives_achievability_input = [a.strip() for a in alternatives_achievability_input]
    alternatives_effect_input = alternatives_achievability_input.copy()  
    alternatives_effect_input = [a.strip() for a in alternatives_effect_input]
        
    
    criteria_alternative_achievability_relationship = {}
    for index, achievability_criterion in enumerate(criteria_achievability_input):
        prompt = f"Enter alternatives related to {achievability_criterion} separated by comma"
        
        alternatives_input = st.text_input(prompt, 'Select from alternatives, for example, projects', key=f"ach_alt_achiev_{index}").split(',')
        criteria_alternative_achievability_relationship[achievability_criterion] = [a.strip() for a in alternatives_input] 
    
    
    criteria_alternative_effect_relationship = criteria_alternative_achievability_relationship.copy()
    
    criteria_achievability_grades_input = st.text_area('Enter grades for pairwise comparisons of criteria on Achievability, separated by comma')
    criteria_effect_grades_input = st.text_area('Enter grades for pairwise comparisons of criteria on Effect, separated by comma')
    
    
    criteria_achievability_grades = input_to_grades_dict(criteria_achievability_grades_input, criteria_achievability_input)
    criteria_effect_grades = input_to_grades_dict(criteria_effect_grades_input, criteria_effect_input)
    

    
    alternative_achievability_grades = {criterion: {} for criterion in criteria_achievability_input}
    alternative_effect_grades = {criterion: {} for criterion in criteria_effect_input}
    
    
    for achievability_criterion in criteria_achievability_input:
        related_achievability_alternatives = criteria_alternative_achievability_relationship[achievability_criterion]
        
        unique_key_achievability = f"grades_achievability_{achievability_criterion}"
        
        achievability_grades_input = st.text_area(f"Enter achievability grades for all alternatives related to {achievability_criterion} separated by comma", key=unique_key_achievability)
        
        alternative_achievability_grades[achievability_criterion] = input_to_grades_dict_for_alternatives(achievability_grades_input, related_achievability_alternatives)
     
        

    for effect_criterion in criteria_effect_input:
        related_effect_alternatives = criteria_alternative_effect_relationship[effect_criterion]
        unique_key_effect = f"grades_effect_{effect_criterion}"
        effect_grades_input = st.text_area(f"Enter effect grades for all alternatives related to {effect_criterion} separated by comma", key=unique_key_effect)
    
        alternative_effect_grades[effect_criterion] = input_to_grades_dict_for_alternatives(effect_grades_input, related_effect_alternatives)

   
    if st.button('Calculate Weights'):
        try:
            synthetic_alternative_achievability_weights = {}  
            synthetic_alternative_effect_weights = {}  
            
                        
            criteria_achievability_matrix = create_pairwise_comparison_matrix_from_grades(criteria_achievability_input, criteria_achievability_grades)
            criteria_effect_matrix = create_pairwise_comparison_matrix_from_grades(criteria_effect_input, criteria_effect_grades)   
            criteria_achievability_vector = calculate_priority_vector(criteria_achievability_matrix) 
            criteria_effect_vector = calculate_priority_vector(criteria_effect_matrix) 
            
    
            for achievability_criterion, achievability_alternatives in criteria_alternative_achievability_relationship.items():
                achievability_matrix = create_pairwise_comparison_matrix_from_grades(achievability_alternatives, alternative_achievability_grades[achievability_criterion])
                alternative_achievability_vector = calculate_priority_vector(achievability_matrix)  
    
                index_achievability = criteria_achievability_input.index(achievability_criterion)
                criteria_weight_for_achievability = criteria_achievability_vector[index_achievability]
    
                for i, achievability_alternative in enumerate(achievability_alternatives):
                    synthetic_alternative_achievability_weights[achievability_alternative] = synthetic_alternative_achievability_weights.get(achievability_alternative, 0) + alternative_achievability_vector[i] * criteria_weight_for_achievability
                    
            for effect_criterion, effect_alternatives in criteria_alternative_effect_relationship.items():
                effect_matrix = create_pairwise_comparison_matrix_from_grades(effect_alternatives, alternative_effect_grades[effect_criterion])
                alternative_effect_vector = calculate_priority_vector(effect_matrix)  
    
                index_effect = criteria_effect_input.index(effect_criterion)
                criteria_weight_for_effect = criteria_effect_vector[index_effect]
    
                for i, effect_alternative in enumerate(effect_alternatives):
                    synthetic_alternative_effect_weights[effect_alternative] = synthetic_alternative_effect_weights.get(effect_alternative, 0) + alternative_effect_vector[i] * criteria_weight_for_effect
            
        
            
            consolidated_achievability_weights = pd.Series(synthetic_alternative_achievability_weights).sort_values(ascending=False)
            consolidated_effect_weights = pd.Series(synthetic_alternative_effect_weights).sort_values(ascending=False)
            
            
            st.table(consolidated_achievability_weights.reset_index().rename(columns={'index': 'Alternative', 0: 'Synthetic Achievability Weight'}))
            st.table(consolidated_effect_weights.reset_index().rename(columns={'index': 'Alternative', 0: 'Synthetic Effect Weight'}))
            
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))  
    
            
            
            median_achievability = consolidated_achievability_weights.median()
            median_effect = consolidated_effect_weights.median()
            ax1.axvline(x=median_achievability, color='red', linestyle='--', label='Median Achievability')
            ax1.axhline(y=median_effect, color='red', linestyle='--', label='Median Effect')
            
            
            ax1.fill_between([median_achievability, max(consolidated_achievability_weights.values)], median_effect, max(consolidated_effect_weights.values), color='#78C850', alpha=0.3)
            
            ax1.fill_between([0, median_achievability], median_effect, max(consolidated_effect_weights.values), color='#FDB147', alpha=0.3)
            ax1.fill_between([median_achievability, max(consolidated_achievability_weights.values)], 0, median_effect, color='#FDB147', alpha=0.3)
            
            ax1.fill_between([0, median_achievability], 0, median_effect, color='#E57373', alpha=0.3)
            
            
            for alternative, achievability_weight in zip(consolidated_achievability_weights.index, consolidated_achievability_weights):
                effect_weight = consolidated_effect_weights[alternative]
                ax1.scatter(achievability_weight, effect_weight, color='blue')
                ax1.annotate(alternative, (achievability_weight, effect_weight), textcoords="offset points", xytext=(10,10), ha='right')
            
            ax1.set_xlabel('Synthetic Alternative Achievability Weights')
            ax1.set_ylabel('Synthetic Alternative Effect Weights')
            ax1.set_title('Scatter Plot of Synthetic Weights')
            ax1.legend()
            
            
            ax2.axis('off')  
            
            
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