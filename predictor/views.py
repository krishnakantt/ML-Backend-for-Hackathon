from rest_framework.decorators import api_view
from rest_framework.response import Response
from .ml_model import model, customer_type_map, product_name_map
import numpy as np

@api_view(["POST"])
def predictor(request):
    if model is None:
        return Response({"error": "ML model is not loaded"}, status=500)
    
    try:
        data = request.data

        features = np.array([[
            float(data.get("Unit_Cost", 0)),
            int(data.get("Order_Quantity", 0)),
            float(data.get("Unit_Sale_Price", 0)),
            float(data.get("Total_Cost", 0)),
            float(data.get("Revenue", 0)),
            int(data.get("Order_Month", 1)),
            int(data.get("Year", 2025)),
            customer_type_map.get(data.get("Customer_Type", ""), 0),
            product_name_map.get(data.get("Product_Name", ""), 0)
        ]])

        y_pred = model.predict(features)[0]
        return Response({"prediction": float(y_pred)})

    except Exception as e:
        return Response({"error": str(e)}, status=400)