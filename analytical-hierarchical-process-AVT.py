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
import networkx as nx
import matplotlib.ticker as ticker

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

def draw_hierarchy_diagram(criteria_nodes, alternatives_nodes, criteria_alternative_relationship_diagram):
    G = nx.DiGraph()

    
    for criterion, related_alternatives in criteria_alternative_relationship_diagram.items():
        for alternative in related_alternatives:
            G.add_edge(alternative, criterion)

    
    goal = "Main goal"
    for criterion in criteria_nodes:
        G.add_edge(criterion, goal)

    
    pos = {}
    pos[goal] = (0.5, 1)  

    
    base_height_criteria = 0.7
    height_variation = 0.1
    criteria_spacing = 1.0 / (len(criteria_nodes) + 1)
    for i, criterion in enumerate(criteria_nodes):
        pos[criterion] = (criteria_spacing * (i + 1), base_height_criteria - (i % 2) * height_variation)

    base_height_alternatives = 0.4
    alternatives_spacing = 1.0 / (len(alternatives_nodes) + 1)
    for i, alternative in enumerate(alternatives_nodes):
        pos[alternative] = (alternatives_spacing * (i + 1), base_height_alternatives - (i % 2) * height_variation)

    for node in G.nodes():
        if node not in pos:
            print(f"Assigning default position to node: {node}")  
            pos[node] = (0.5, 0.5)  

    # Drawing the graph
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=2000, 
            node_color='skyblue', font_weight='bold', ax=ax,
            arrowstyle='->', arrowsize=15)

    plt.title("Create and update your hierarchy Process diagram with the fields below")
    return fig

st.set_page_config(layout="wide")

def app():
    
    st.title('Analytical Hierarchical Process with Criteria-Alternative Relationships')
    
    st.markdown('''
    ### Welcome to a Simplified Analytical Hierarchy Process (AHP) Tool, a user-friendly application inspired by the groundbreaking work of mathematician Thomas L. Saaty. This tool provides a straightforward approach to the AHP, allowing you to create your own AHP diagrams, define criteria and alternatives, and grade each based on achievability and effect. You can update the diagram below by filling out the form.
        ''')
        
       
    criteria_nodes = ['Criteria 1', 'Criteria 2']  
    alternatives_nodes = ['Alternative 1', 'Alternative 2']  
    criteria_alternative_relationship_diagram = {}  
    
    
    diagram_placeholder = st.empty()
    
    
    fig = draw_hierarchy_diagram(criteria_nodes, alternatives_nodes, criteria_alternative_relationship_diagram)
    diagram_placeholder.pyplot(fig)
    
    st.markdown('''
    This application deviates slightly from the traditional method of grading pairwise comparisons. Instead, you can directly assign grades to individual criteria and alternatives, on the basis of their achievability and effect for the related node. The app then uses these grades to conduct pairwise comparisons automatically, ensuring completely consistent matrices, without the need for consistency metrics.            
     ### How It Works (interactive form at the bottom of page)
     1. **Design Your AHP Diagram:** Start by laying out the structure of your decision-making process.
     2. **Specify Criteria and Alternatives:** Define the elements of your decision matrix.
     3. **Grade Each Element:** Assign a grade to each criterion and alternative, focusing on achievability and effect. Please note, grades for multiple criteria and alternatives should be comma-separated, and in the same order as registered. The amount of grades should also match the elements being graded.
     
     To facilitate the input process, you can prepare your data using worksheets, making registration straightforward. Although currently grades are registered manually, and depending on user feedback on needs, future models might allow direct uploads of CSV or XLSX files, to simplify modelling more complex processes.
     
     ### Features
     - **Update Diagram:** A dedicated button allows you to refresh the diagram as needed during the grading process.
     - **Calculate Weights:** Once your data is input, calculate the weights to see how criteria and alternatives stack up.
     - **Visualize Priorities:** The app generates a scatter plot for easy visualization of priorities, based on your predefined criteria.
     
     This tool is perfect for anyone looking to apply the AHP in a simplified and intuitive manner. Whether for academic, personal, or professional decisions, this app makes the AHP accessible and straightforward.
     
     For more on the analytical hierarchy process as conceived by Thomas L. Saaty, see [this article](https://www.sciencedirect.com/science/article/pii/0270025587904738).
     
     ### Feedback
     Have questions or suggestions? Feel free to contact me at [alberto@vthoresen.no](mailto:alberto@vthoresen.no) or visit [GitHub repository](https://github.com/albertovth/analyitical-hierarchy-process) for more information.
     
     ''')            
    
    st.markdown('''
    ### Interactive Process Design and Evaluation Form
    You will find the buttons to update the process design and/or calculate weights and produce priority scatter plot below
    ''')
    
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
    
    
    criteria_nodes = criteria_achievability_input.copy()
    
    alternatives_nodes = alternatives_achievability_input.copy()
       
    
    
    criteria_alternative_relationship_diagram = criteria_alternative_achievability_relationship.copy()

    if st.button('Update process design diagram (image at top of page)'):
    
        criteria_nodes = criteria_achievability_input  # Assuming these variables are updated with user inputs
        alternatives_nodes = alternatives_achievability_input
    

    
        fig = draw_hierarchy_diagram(criteria_nodes, alternatives_nodes, criteria_alternative_relationship_diagram)
    
        diagram_placeholder.pyplot(fig)
       
    
    if st.button('Calculate Weights and generate priority scatter plot'):
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
            
            
            
            achievability_criteria_weights_df = pd.DataFrame({
                'Criteria': criteria_achievability_input,
                'Criteria Achievability Weight (%)': criteria_achievability_vector * 100
            })
            
            
            achievability_criteria_weights_df = achievability_criteria_weights_df.sort_values(by='Criteria Achievability Weight (%)', ascending=False)
            
            
            st.table(achievability_criteria_weights_df)
            
            
            
            effect_criteria_weights_df = pd.DataFrame({
                'Criteria': criteria_effect_input,
                'Criteria Effect Weight (%)': criteria_effect_vector * 100
            })
            
            
            effect_criteria_weights_df = effect_criteria_weights_df.sort_values(by='Criteria Effect Weight (%)', ascending=False)
            
            
            st.table(effect_criteria_weights_df)
            
        
            
            consolidated_achievability_weights = pd.Series(synthetic_alternative_achievability_weights).sort_values(ascending=False)
            consolidated_effect_weights = pd.Series(synthetic_alternative_effect_weights).sort_values(ascending=False)
            
            consolidated_achievability_weights_percent = consolidated_achievability_weights.copy()*100 
            consolidated_effect_weights_percent = consolidated_effect_weights.copy()*100
            
            
            st.table(consolidated_achievability_weights_percent.reset_index().rename(columns={'index': 'Alternative', 0: 'Synthetic Achievability Weight'}))
            st.table(consolidated_effect_weights_percent.reset_index().rename(columns={'index': 'Alternative', 0: 'Synthetic Effect Weight'}))
            
            
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
            
            
            ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
            ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
            
            
            ax1.set_xlabel('Synthetic Alternative Achievability Weights (%)')
            ax1.set_ylabel('Synthetic Alternative Effect Weights (%)')
            
            
            ax1.set_title('Scatter Plot of Synthetic Weights (Percentages)')
            
            
            ax1.legend()

            
            
            ax2.axis('off')  
            
            
            bgcolor = ['#78C850', '#FDB147', '#E57373']
            text = ['Top Priority - Light Green', 'Priority - Light Yellow', 'Not Priority - Light Red']
            y_positions = [0.6, 0.5, 0.4]
            
            for bg, txt, y in zip(bgcolor, text, y_positions):
                ax2.text(0.5, y, txt, horizontalalignment='center', verticalalignment='center', fontsize=10, bbox=dict(facecolor=bg, alpha=0.5, edgecolor='none'))
            
            plt.tight_layout()
            st.pyplot(fig)
            
        
            criteria_nodes = criteria_achievability_input  
            alternatives_nodes = alternatives_achievability_input
        

        
            fig = draw_hierarchy_diagram(criteria_nodes, alternatives_nodes, criteria_alternative_relationship_diagram)
        
            diagram_placeholder.pyplot(fig)
           
            
    
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == '__main__':
    app()
