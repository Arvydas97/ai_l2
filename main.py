import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


def draw():
    # Visualize these universes and membership functions
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

    ax0.plot(x_height, height_lo, 'g', linewidth=1.5, label='Small')
    ax0.plot(x_height, height_md, 'b', linewidth=1.5, label='Medium')
    ax0.plot(x_height, height_hi, 'r', linewidth=1.5, label='Large')
    ax0.set_title('Height, kg')
    ax0.legend()

    ax1.plot(x_weight, weight_lo, 'g', linewidth=1.5, label='Small')
    ax1.plot(x_weight, weight_md, 'b', linewidth=1.5, label='Medium')
    ax1.plot(x_weight, weight_hi, 'm', linewidth=1.5, label='Large')
    ax1.plot(x_weight, weight_vh, 'r', linewidth=1.5, label='Very large')
    ax1.set_title('Weight, cm')
    ax1.legend()

    ax2.plot(x_age, age_lo, 'g', linewidth=1.5, label='Young')
    ax2.plot(x_age, age_md, 'b', linewidth=1.5, label='Middle-aged')
    ax2.plot(x_age, age_hi, 'r', linewidth=1.5, label='Old')
    ax2.set_title('Age, yrs')
    ax2.legend()

    ax3.plot(x_conclusion, con_lo, 'g', linewidth=1.5, label='Lean')
    ax3.plot(x_conclusion, con_md, 'b', linewidth=1.5, label='Normal')
    ax3.plot(x_conclusion, con_hi, 'r', linewidth=1.5, label='Fat')
    ax3.set_title('Body status')
    ax3.legend()

    # Turn off top/right axes
    for ax in (ax0, ax1, ax2, ax3):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()


def calcLevels(height, weight, age):
    height_level_lo = fuzz.interp_membership(x_height, height_lo, height)
    height_level_md = fuzz.interp_membership(x_height, height_md, height)
    height_level_hi = fuzz.interp_membership(x_height, height_hi, height)

    weight_level_lo = fuzz.interp_membership(x_weight, weight_lo, weight)
    weight_level_md = fuzz.interp_membership(x_weight, weight_md, weight)
    weight_level_hi = fuzz.interp_membership(x_weight, weight_hi, weight)
    weight_level_vh = fuzz.interp_membership(x_weight, weight_vh, weight)

    age_level_lo = fuzz.interp_membership(x_age, age_lo, age)
    age_level_md = fuzz.interp_membership(x_age, age_md, age)
    age_level_hi = fuzz.interp_membership(x_age, age_hi, age)

    prob_lean = calcLean(height_level_md, weight_level_lo, age_level_hi, height_level_hi, weight_level_md)
    prob_norm = calcNormal(height_level_lo, height_level_md, height_level_hi, weight_level_lo,
                           weight_level_md, weight_level_hi, weight_level_vh, age_level_lo, age_level_md)
    prob_fat = calcFat(height_level_lo, height_level_md, height_level_hi, weight_level_hi,
                       weight_level_vh, age_level_md, age_level_hi)

    return prob_lean, prob_norm, prob_fat


def drawProbs(prob_lean, prob_norm, prob_fat):
    sold0 = np.zeros_like(x_conclusion)
    # Visualizerezultsofmembershipactivit
    fig, ax0 = plt.subplots(figsize=(8, 3))
    ax0.set_title('Body status')

    ax0.fill_between(x_conclusion, sold0, prob_lean, facecolor='aqua', alpha=0.7)
    ax0.plot(x_conclusion, con_lo, 'turquoise', linewidth=1, linestyle='--', label='Lean')

    ax0.fill_between(x_conclusion, sold0, prob_norm, facecolor='aquamarine', alpha=0.7)
    ax0.plot(x_conclusion, con_md, 'blue', linewidth=1, linestyle='--', label='Normall')

    ax0.fill_between(x_conclusion, sold0, prob_fat, facecolor='lightgreen', alpha=0.7)
    ax0.plot(x_conclusion, con_hi, 'green', linewidth=1, linestyle='--', label='Fat')
    ax0.legend()

    for ax in (ax0,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()


def aggregation(prob_lean, prob_norm, prob_fat):
    aggregated = np.fmax(prob_lean, np.fmax(prob_norm, prob_fat))
    sold0 = np.zeros_like(x_conclusion)
    # Calculatedefuzzifiedresult
    sold_centroid = fuzz.defuzz(x_conclusion, aggregated, 'centroid')
    sold_bisector = fuzz.defuzz(x_conclusion, aggregated, 'bisector')
    sold_mom = fuzz.defuzz(x_conclusion, aggregated, 'mom')  # meanofmaximum
    sold_som = fuzz.defuzz(x_conclusion, aggregated, 'som')  # minofmaximum
    sold_lom = fuzz.defuzz(x_conclusion, aggregated, 'lom')  # maxofmaximum

    sold_activation = fuzz.interp_membership(x_conclusion, aggregated, sold_centroid)  # forplot

    # Visualizethis
    fig, ax0 = plt.subplots(figsize=(8, 3))
    ax0.plot(x_conclusion, con_lo, 'turquoise', linewidth=1, linestyle='--', label="Lean")
    ax0.plot(x_conclusion, con_md, 'blue', linewidth=1, linestyle='--', label="Normall")
    ax0.plot(x_conclusion, con_hi, 'green', linewidth=1, linestyle='--', label="Fat")
    ax0.fill_between(x_conclusion, sold0, aggregated, facecolor='aquamarine', alpha=0.7)
    ax0.plot([sold_centroid, sold_centroid], [0, sold_activation], 'purple', linewidth=1.5, alpha=0.9, label="Centroid")
    ax0.plot([sold_bisector, sold_bisector], [0, sold_activation], 'brown', linewidth=1.5, alpha=0.9, label="Bisector")
    ax0.plot([sold_mom, sold_mom], [0, 0.4], 'red', linewidth=1.5, alpha=0.9, label="MOM")
    ax0.legend()

    ax0.set_title('Aggregated results')
    # Turnofftop/rightaxes
    for ax in (ax0,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()

    print("Defuzzification: Centroid")
    print(np.round(sold_centroid, 2))
    print("Defuzzification: Bisector")
    print(np.round(sold_bisector, 2))
    print("Defuzzification: Mean of maximum")
    print(np.round(sold_mom, 2))
    print("Defuzzification: Min of maximum")
    print(np.round(sold_som, 2))
    print("Defuzzification: Max of maximum")
    print(np.round(sold_lom, 2))


def calcLean(height_level_md, weight_level_lo, age_level_hi, height_level_hi, weight_level_md):
    # * Jei ūgis vidutinis, ir svoris mažas, ir amžius senas- liesas
    # * Jei ūgis didelis, ir svoris mažas arba vidutinis- lieknas
    active_rule1 = min(height_level_md, weight_level_lo, age_level_hi)
    active_rule2 = min(height_level_hi, max(weight_level_lo, weight_level_md))

    probability = max(active_rule1, active_rule2)
    sold_activation_lo = np.fmin(probability, con_lo)
    print("Answer for lean: {ans:}".format(ans=probability))
    return sold_activation_lo


def calcNormal(height_level_lo, height_level_md, height_level_hi, weight_level_lo,
               weight_level_md, weight_level_hi, weight_level_vh, age_level_lo, age_level_md):
    # * Jei ūgis mažas, ir svoris mažas arba vidutinis- vidutinis
    # * Jei ūgis vidutinis, ir svoris mažas, ir amžius vidutinis arba jaunas- vidutinis
    # * Jei ūgis vidutinis, ir svoris vidutinis- vidutinis
    # * Jei ūgis vidutinis arba didelis, ir svoris didelis- vidutinis
    # * Jei ūgis vidutinis, ir svoris labai didelis, ir amžius jaunas- vidutinis
    # * Jei ūgis didelis, ir svoris labai didelis, ir amžius jaunas arba vidutinis- vidutinis
    active_rule1 = min(height_level_lo, max(weight_level_lo, weight_level_md))
    active_rule2 = min(height_level_md, weight_level_lo, max(age_level_lo, age_level_md))
    active_rule3 = min(height_level_md, weight_level_md)
    active_rule4 = min(max(height_level_md, height_level_hi), weight_level_hi)
    active_rule5 = min(height_level_md, weight_level_vh, age_level_lo)
    active_rule6 = min(height_level_md, weight_level_vh, max(age_level_lo, age_level_md))

    probability = max(active_rule1, active_rule2, active_rule3, active_rule4, active_rule5, active_rule6)
    sold_activation_md = np.fmin(probability, con_md)
    print("Answer for normal: {ans:}".format(ans=probability))
    return sold_activation_md


def calcFat(height_level_lo, height_level_md, height_level_hi, weight_level_hi,
            weight_level_vh, age_level_md, age_level_hi):
    # * Jei ūgis mažas, ir svoris didelis arba labai didelis- nutukęs
    # * Jei ūgis vidutinis, ir svoris labai didelis, ir amžius senas arba vidutinis- nutukęs
    # * Jei ūgis didelis, ir svoris labai didelis, ir amžius senas- nutukęs
    active_rule1 = min(height_level_lo, max(weight_level_hi, weight_level_vh))
    active_rule2 = min(height_level_md, weight_level_vh, max(age_level_md, age_level_hi))
    active_rule3 = min(height_level_hi, weight_level_vh, age_level_hi)

    probability = max(active_rule1, active_rule2, active_rule3)
    sold_activation_hi = np.fmin(probability, con_hi)
    print("Answer for fat: {ans:}".format(ans=probability))
    return sold_activation_hi


if __name__ == '__main__':
    x_height = np.arange(15, 100, 1)
    x_weight = np.arange(1, 70, 1)
    x_age = np.arange(0, 15, 1)
    x_conclusion = np.arange(0, 100, 1)

    # Generate fuzzy membership functions
    height_lo = fuzz.trapmf(x_height, [15, 15, 25, 35])
    height_md = fuzz.trimf(x_height, [25, 40, 60])
    height_hi = fuzz.trapmf(x_height, [45, 70, 100, 100])

    weight_lo = fuzz.trimf(x_weight, [1, 1, 7])
    weight_md = fuzz.trapmf(x_weight, [5, 15, 20, 25])
    weight_hi = fuzz.trapmf(x_weight, [18, 30, 45, 60])
    weight_vh = fuzz.trapmf(x_weight, [45, 60, 70, 70])

    age_lo = fuzz.trimf(x_age, [0, 0, 2])
    age_md = fuzz.trimf(x_age, [1, 6, 10])
    age_hi = fuzz.trapmf(x_age, [8, 12, 15, 15])

    con_lo = fuzz.trimf(x_conclusion, [0, 0, 35])
    con_md = fuzz.trimf(x_conclusion, [20, 50, 75])
    con_hi = fuzz.trimf(x_conclusion, [60, 100, 100])

    draw()

    test_height = 40
    test_weight = 17
    test_age = 5

    p_lean, p_norm, p_fat = calcLevels(test_height, test_weight, test_age)
    drawProbs(p_lean, p_norm, p_fat)
    aggregation(p_lean, p_norm, p_fat)
