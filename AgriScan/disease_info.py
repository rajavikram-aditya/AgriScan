# disease_info.py
# Dictionary mapping every class name to a "description" and "treatment" string.

disease_info = {
    # Existing Potato Classes
    "Potato___healthy": {
        "description": "The potato plant appears healthy with no visible signs of disease.",
        "treatment": "Maintain optimal watering and fertilizer practices to keep the plant healthy."
    },
    "Potato___Early_blight": {
        "description": "Early blight is a fungal disease caused by Alternaria solani. It manifests as dark brown to black spots with concentric rings on the lower, older leaves.",
        "treatment": "Remove and destroy infected leaves. Apply copper-based fungicides or other broad-spectrum fungicides. Ensure proper spacing for air circulation and avoid overhead watering."
    },
    "Potato___Late_blight": {
        "description": "Late blight is a destructive disease caused by the oomycete Phytophthora infestans. It shows as water-soaked spots on leaves that enlarge rapidly into brown, purplish-black necrotic lesions.",
        "treatment": "Apply protective fungicides containing chlorothalonil, mancozeb, or copper. Destroy infected plants immediately to prevent spread. Avoid damp and cool conditions if possible."
    },
    
    # Tomato Classes
    "Tomato___Bacterial_spot": {
        "description": "Bacterial spot is caused by Xanthomonas bacteria, resulting in small, dark, water-soaked spots on leaves and scabby lesions on fruits.",
        "treatment": "Use disease-free seeds and transplants. Apply copper fungicides. Avoid overhead watering and rotate crops."
    },
    "Tomato___Early_blight": {
        "description": "Early blight is a fungal disease (Alternaria linariae) causing dark, concentric rings on older leaves and stem lesions.",
        "treatment": "Remove infected plant debris. Ensure good air circulation. Apply fungicides such as chlorothalonil or copper early in the season."
    },
    "Tomato___Late_blight": {
        "description": "Late blight (Phytophthora infestans) causes large, dark brown spots with irregular edges on leaves and stems, often rapidly destroying the plant.",
        "treatment": "Remove and destroy infected plants. Apply protective fungicides containing chlorothalonil or copper. Avoid wet leaves by watering at the base."
    },
    "Tomato___Leaf_Mold": {
        "description": "Leaf mold (Passalora fulva) primarily affects tomatoes in high humidity, showing pale green or yellow spots on upper leaves and olive-green mold on the undersides.",
        "treatment": "Improve air circulation and reduce humidity. Water at the base of plants. Use resistant varieties and apply appropriate fungicides if severe."
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Caused by the fungus Septoria lycopersici, it produces numerous small, circular spots with dark borders and lighter centers on lower leaves.",
        "treatment": "Remove infected leaves. Mulch around the base to prevent soil splashing. Apply fungicides like chlorothalonil early in the infection."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Two-spotted spider mites are tiny arachnids that suck sap from leaves, causing stippling, yellowing, and fine webbing on the plant.",
        "treatment": "Spray plants with a strong stream of water to dislodge mites. Use insecticidal soap, neem oil, or horticultural oils. Encourage natural predators like ladybugs."
    },
    "Tomato___Target_Spot": {
        "description": "Target spot is a fungal disease (Corynespora cassiicola) that causes dark brown spots with light brown centers, often with a target-like appearance on leaves.",
        "treatment": "Improve air flow by pruning lower leaves. Apply fungicides containing chlorothalonil or copper. Avoid overhead irrigation."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "This viral disease is transmitted by whiteflies. Leaves become yellow, curled upwards, and stunted, severely reducing fruit yield.",
        "treatment": "Remove infected plants immediately. Control whitefly populations using yellow sticky traps, insecticidal soaps, or neem oil. Plant resistant varieties."
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "A highly contagious viral disease that causes mottled, distorted leaves and stunted growth. It is often spread through seeds, tools, and human contact.",
        "treatment": "There is no cure; infected plants must be destroyed. Wash hands and sterilize tools frequently. Plant resistant varieties."
    },
    "Tomato___healthy": {
        "description": "The tomato plant appears healthy with no visible signs of disease or pest infestation.",
        "treatment": "Continue regular maintenance, adequate watering at the base, and balanced fertilization to keep the plant thriving."
    },

    # Corn Classes
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Gray leaf spot is a fungal disease (Cercospora zeae-maydis) that forms rectangular, pale brown to gray lesions restricted by leaf veins.",
        "treatment": "Use resistant hybrids. Implement crop rotation and tillage to reduce fungal inoculum in the crop residue. Apply foliar fungicides if disease pressure is high."
    },
    "Corn_(maize)___Common_rust_": {
        "description": "Common rust is caused by the fungus Puccinia sorghi. It produces small, reddish-brown pustules on both upper and lower leaf surfaces.",
        "treatment": "Plant rust-resistant corn varieties. Fungicide applications can be effective if applied early when pustules first appear, especially in susceptible varieties."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "Caused by Exserohilum turcicum, Northern Leaf Blight results in large, cigar-shaped, grayish-green to tan lesions on the leaves.",
        "treatment": "Grow resistant hybrids. Practice crop rotation and bury infected residue. Fungicides may be used if symptoms appear early in the season."
    },
    "Corn_(maize)___healthy": {
        "description": "The corn plant appears healthy with no visible signs of foliar diseases.",
        "treatment": "Maintain good agronomic practices including proper fertilization, watering, and weed control."
    },

    # Rice Classes
    "Rice___Bacterial_Leaf_Blight": {
        "description": "Bacterial leaf blight causes water-soaked streaks on leaves, which enlarge and turn yellowish-white, often leading to rapid wilting.",
        "treatment": "Use disease-free seeds and resistant varieties. Avoid excessive nitrogen application. Apply copper-based bactericides if necessary."
    },
    "Rice___Brown_Spot": {
        "description": "Brown spot is a fungal disease (Bipolaris oryzae) that causes numerous small, oval to circular brown spots on leaves, often indicating poor soil fertility.",
        "treatment": "Improve soil fertility with proper fertilization, especially nitrogen and silicon. Treat seeds with fungicides before planting. Ensure good water management."
    },
    "Rice___Healthy_Rice_Leaf": {
        "description": "The rice plant appears healthy and vigorous, free from major diseases.",
        "treatment": "Maintain optimal water levels, proper fertilization, and monitor regularly for early signs of pests or disease."
    },
    "Rice___Leaf_Blast": {
        "description": "Rice blast (Magnaporthe oryzae) is a highly destructive fungal disease causing diamond-shaped lesions with gray centers and brown borders on leaves.",
        "treatment": "Plant resistant varieties. Apply nitrogen in split doses rather than all at once. Maintain consistent flooding and use appropriate systemic fungicides if required."
    },
    "Rice___Leaf_scald": {
        "description": "Leaf scald (Microdochium oryzae) starts as water-soaked lesions at the leaf tips, progressing downwards and turning grayish-white.",
        "treatment": "Use balanced fertilization avoiding excess nitrogen. Apply systemic fungicides and maintain good field sanitation."
    },
    "Rice___Sheath_Blight": {
        "description": "Sheath blight causes oval, greenish-gray water-soaked spots on the leaf sheaths near the water level, stunting plant growth.",
        "treatment": "Maintain appropriate plant spacing for ventilation. Apply fungicides such as azoxystrobin or propiconazole at the early booting stage."
    },

    # Wheat Classes
    "Wheat___BlackPoint": {
        "description": "Black point is a fungal disease that causes a dark discoloration at the embryo end of the wheat kernel.",
        "treatment": "Apply fungicides at the heading stage if weather conditions are humid. Avoid late harvesting."
    },
    "Wheat___FusariumFootRot": {
        "description": "Fusarium foot rot causes browning and rotting of the lower stem internodes, leading to premature ripening and white heads.",
        "treatment": "Practice crop rotation with non-cereal crops. Delay autumn sowing and use treated seeds."
    },
    "Wheat___HealthyLeaf": {
        "description": "The wheat plant appears healthy, exhibiting strong growth and no signs of rust or other diseases.",
        "treatment": "Ensure timely irrigation, proper crop rotation, and balanced nutrient application to sustain crop health."
    },
    "Wheat___LeafBlight": {
        "description": "Leaf blight causes small, yellowish-brown spots that expand into large, irregular necrotic areas on the leaves.",
        "treatment": "Use resistant varieties. Apply foliar fungicides early in the disease cycle, especially in wet conditions."
    },

    # Cotton Classes
    "Cotton___bacterial_blight": {
        "description": "Bacterial blight in cotton causes angular, water-soaked leaf spots that turn brown or black, leading to defoliation and boll rot.",
        "treatment": "Use resistant cotton varieties and disease-free seeds. Plow under crop debris after harvest to reduce bacterial survival in the soil."
    },
    "Cotton___curl_virus": {
        "description": "Cotton leaf curl virus is transmitted by whiteflies, causing leaves to curl upward or downward and severely stunting plant growth.",
        "treatment": "Control whitefly populations with appropriate insecticides or neem oil. Remove and destroy infected plants immediately. Use resistant varieties."
    },
    "Cotton___fussarium_wilt": {
        "description": "Fusarium wilt is a soil-borne fungal disease causing yellowing, wilting, and necrosis of leaves, accompanied by dark vascular discoloration in the stem.",
        "treatment": "Plant resistant varieties and rotate crops with non-hosts for several years. Ensure proper soil drainage and balanced potassium fertilization."
    },
    "Cotton___healthy": {
        "description": "The cotton plant appears healthy with no visible signs of disease.",
        "treatment": "Maintain optimal watering and nutrient management. Monitor regularly for early signs of pests like whiteflies or aphids."
    },

    # Sugarcane Classes
    "Sugarcane___Healthy": {
        "description": "The sugarcane plant appears healthy with no signs of disease.",
        "treatment": "Ensure adequate irrigation and fertilization, particularly nitrogen and potassium, to maintain robust growth."
    },
    "Sugarcane___Mosaic": {
        "description": "Sugarcane mosaic virus causes mottling and yellowing of the leaves in irregular patterns, stunting growth and reducing sugar yield.",
        "treatment": "Use certified virus-free seed cane. Control aphid vectors using insecticidal soaps and remove infected plants from the field."
    },
    "Sugarcane___RedRot": {
        "description": "Red rot (Colletotrichum falcatum) is one of the most destructive diseases, causing yellowing of leaves and internal reddening of the stalk with white transverse bands.",
        "treatment": "Plant resistant varieties and use healthy setts for planting. Ensure good field drainage and practice crop rotation."
    },
    "Sugarcane___Rust": {
        "description": "Sugarcane rust forms elongated, reddish-brown pustules on the leaves, leading to premature leaf death and reduced photosynthesis.",
        "treatment": "Plant rust-resistant cultivars. Apply foliar fungicides if the infection is detected early in susceptible varieties."
    },
    "Sugarcane___Yellow": {
        "description": "Yellow leaf syndrome, caused by a luteovirus, results in intense yellowing of the leaf midrib on the lower surface, spreading to the blade.",
        "treatment": "Use pathogen-free planting material. Control sugarcane aphids which transmit the virus. Eliminate infected volunteer plants."
    }
}
