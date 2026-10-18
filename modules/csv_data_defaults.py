import os

DEFAULT_NHANES_PERSONAS_CSV = '''persona_id,name,gender,age,bmi,systolic_bp,diastolic_bp,fasting_glucose_mmol,hba1c_pct,bun_mmol,conditions,allergies,special_diets,dietary_habits,description
USER_CLINICAL_001,Bệnh nhân Phenylketon niệu (PKU) — Chống chỉ định Tuyệt đối Aspartame,Female,26,21.5,115,75,4.8,5.1,4.5,"PKU, Inborn_Errors_of_Metabolism","aspartame, phenylalanine","Phenylalanine-Free Diet, Low-Protein Diet",Never drinks diet soda with artificial sweeteners,Bệnh nhân mắc chứng rối loạn chuyển hóa Phenylketon niệu di truyền; gan không thể tổng hợp enzym phenylalanine hydroxylase. Bắt buộc kiêng tuyệt đối nguồn Phenylalanine và Aspartame (INS 951).
USER_CLINICAL_002,Bệnh nhân Tăng huyết áp Độ 2 & Suy tim — Cần kiểm soát nghiêm ngặt Natri & Bột ngọt,Male,58,28.4,155,98,5.6,5.7,6.8,"Hypertension, Cardiovascular_Risk",,"DASH Diet, Low-Sodium Diet (< 1500mg/day)",Rarely adds salt at table; needs low sodium foods,"Huyết áp 155/98 mmHg. Chế độ ăn giới hạn muối natri nghiêm ngặt dưới 1.500 mg/ngày. Cần thận trọng cao với bột ngọt (Mononatri glutamat INS 621, chứa 12.3% natri) và chất bảo quản Natri benzoat (INS 211)."
USER_CLINICAL_003,Trẻ nhỏ 5 tuổi (Cơ địa Hiếu động / ADHD) — Cảnh báo Phẩm màu Azo Southampton,Male,5,15.8,95,60,4.5,4.9,3.8,"Childhood_ADHD_Risk, Children_Under_6",,"Clean Diet, No Synthetic Colors",Frequently consumes snacks and colorful drinks,"Hệ thần kinh và cơ chế thải độc chưa hoàn thiện. EFSA và Thông tư 24/2019/TT-BYT khuyến cáo cảnh báo nguy cơ làm tăng hành vi hiếu động và giảm khả năng tập trung khi tiêu thụ phẩm màu tổng hợp Azo (như Tartrazine INS 102, Sunset Yellow INS 110)."
USER_CLINICAL_004,Phụ nữ Mang thai 3 tháng đầu — Cần ưu tiên Phụ gia Tự nhiên & Vi chất Folic,Female,29,23.2,110,70,4.6,5.0,3.5,Pregnancy_First_Trimester,,"Prenatal Nutrition Diet, High-Folate","Prioritizes natural foods, avoids chemical preservatives","Giai đoạn tạo cơ quan phôi thai nhạy cảm. Cần hạn chế tối đa phụ gia hóa học tổng hợp, chất bảo quản gốc Benzoat và chất tạo ngọt nhân tạo; khuyến khích các chất điều vị và acid tự nhiên (Acid citric INS 330, Lecithin INS 322i)."
USER_CLINICAL_005,"Bệnh nhân Bệnh thận mạn giai đoạn 3 (CKD) — Cần giảm Photpho, Đạm & Phụ gia Natri",Male,62,24.1,138,86,5.9,6.0,14.2,"Chronic_Kidney_Disease_Stage_3, Impaired_Renal_Filtration",,"Renal Diet, Low-Phosphorus, Low-Protein, Low-Sodium",Strictly monitors daily electrolyte intake,Mức lọc cầu thận suy giảm; nồng độ ure huyết (BUN) 14.2 mmol/L. Cần tránh các chất tạo xốp chứa amoni (INS 503ii) và các muối photphat vô cơ gây tích tụ độc tố thận.
USER_CLINICAL_006,"Người có tiền sử Dị ứng Thực phẩm (Đậu nành, Sữa, Trứng, Gluten)",Female,31,21.0,112,72,4.7,5.0,4.0,Severe_Food_Allergies,"gluten, milk, egg, soy, peanut","Allergen-Free Diet, Celiac Diet",Always reads allergen advice on food labels,"Tiền sử sốc phản vệ và nổi mày đay với protein sữa bò, trứng, đậu nành và celiac (dị ứng gluten bột mì). Cần cảnh báo khẩn cấp nếu sản phẩm chứa chất nhũ hóa Lecithin INS 322i nguồn gốc từ đậu nành hoặc có bột mì."
USER_CLINICAL_007,Người trưởng thành Khỏe mạnh — Tiêu dùng Thực phẩm An toàn Thông thường,Male,34,22.4,118,76,5.0,5.2,5.1,Healthy_Adult,,Balanced Diet,Regular balanced diet without dietary restrictions,"Thể trạng hoàn toàn khỏe mạnh, không có bệnh nền mạn tính hay tiền sử dị ứng thực phẩm. Đánh giá phụ gia dựa trên nguyên tắc GMP và tuân thủ ngưỡng ăn vào hàng ngày JECFA ADI."'''

DEFAULT_INCIDENTS_TRAIN_CSV = '''title,product,product_category,hazard,hazard_category
Recall of Butter Cookies due to Undeclared Milk and Egg Allergens,Butter cookies,Cereals and bakery products,Milk and egg allergens,Allergens
Alert: Undeclared Soy Lecithin in Chocolate Chip Cookie Assortment,Chocolate chip cookies,Cereals and bakery products,Undeclared soy lecithin (INS 322i),Allergens
Warning on Instant Noodles: Presence of Undeclared Peanut Residues,Instant noodles,Cereals and bakery products,Peanut traces,Allergens
Withdrawal of Carbonated Cola Drink due to Excessive Sweetener Formulation,Carbonated soft drinks,Non-alcoholic beverages,Excessive aspartame / acesulfame K,Food additives
Recall of Seasoning Powder Packets: High Sodium Concentration Labeling Omission,Seasoning packets,Soups sauces and seasonings,Undeclared high monosodium glutamate,Food additives
Notification: Unauthorized Synthetic Color Tartrazine in Snack Product,Extruded corn snacks,Cereals and bakery products,Unauthorized azo dye Tartrazine E102,Food additives
Precautionary Recall: Milk Bread Rolls with Traces of Sesame and Gluten,Milk bread,Cereals and bakery products,Gluten contamination,Allergens
Safety Alert: Contamination of Soft Drinks with Yeast and Mold,Fruit flavored beverage,Non-alcoholic beverages,Mold and yeast contamination,Microbiological
Recall: Energy Drink with High Level of Benzoic Acid Preservative,Energy drinks,Non-alcoholic beverages,Excessive sodium benzoate INS 211,Food additives
Quality Alert: Canned Carbonated Drink with Can Seam Defect,Canned soft drinks,Non-alcoholic beverages,Metallic foreign matter,Foreign bodies'''

DEFAULT_OPENFOODFACTS_VIETNAM_CSV = '''barcode,product_name,product_category,raw_ingredients,detected_additives,detected_allergens
8934567890123,Bánh quy bơ cao cấp (Butter Cookies),Bánh kẹo đóng gói,"Bột mì, bơ lạt, đường kính, trứng gà, bột sữa béo, INS 322i, INS 500ii, INS 503ii, muối","INS 322i, INS 500ii, INS 503ii","gluten, milk, egg, soy"
8934567890456,Mì ăn liền vị tôm chua cay,Mì và ngũ cốc ăn liền,"Vắt mì: Bột mì, dầu cọ, INS 1420. Gói gia vị: Muối, đường, INS 621, INS 631, INS 627, INS 211, INS 102","INS 621, INS 631, INS 627, INS 211, INS 102, INS 1420","gluten"
8934567890789,Nước giải khát không đường vị Cola,Đồ uống không cồn,"Nước bão hòa CO2, chất điều chỉnh độ acid (INS 330, INS 331iii), chất tạo ngọt tổng hợp (INS 951, INS 950), màu INS 150d","INS 330, INS 331iii, INS 951, INS 950, INS 150d","phenylalanine"
8934567890111,Bánh mì tươi bơ sữa đóng gói,Bánh mì và bột,"Bột mì, sữa tươi, bơ, men, muối, chất bảo quản (INS 282), chất nhũ hóa (INS 471)","INS 282, INS 471","gluten, milk"
8934567890222,Hạt nêm thịt thăn xương ống,Gia vị nêm sẵn,"Muối, đường, chất điều vị (INS 621, INS 631, INS 627), chiết xuất thịt thăn (2%)","INS 621, INS 631, INS 627",""'''

def ensure_default_csv_data(data_dir):
    """
    Tự động đảm bảo 5 tệp dữ liệu CSV chuẩn hóa luôn hiện diện trong thư mục data.
    Nếu bất kỳ tệp nào bị thiếu (do giải nén sót hoặc người dùng chép thiếu file),
    hệ thống sẽ tự động khôi phục ngay lập tức để không bao giờ văng FileNotFoundError.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    files_map = {
        "nhanes_personas.csv": DEFAULT_NHANES_PERSONAS_CSV,
        "incidents_train.csv": DEFAULT_INCIDENTS_TRAIN_CSV,
        "openfoodfacts_vietnam.csv": DEFAULT_OPENFOODFACTS_VIETNAM_CSV,
    }
    
    for filename, content in files_map.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content.strip() + "\n")
            except Exception as e:
                print(f"Không thể tự động khôi phục file {filename}: {e}")
