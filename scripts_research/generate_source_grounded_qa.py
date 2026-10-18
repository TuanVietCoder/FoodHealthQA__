"""Build a source-linked, non-clinical FoodHealthQA draft from authoritative sources."""

import csv
import json
import math
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.pipeline import load_fndds_nutrition


DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "qa_benchmark_600_source_grounded.jsonl")
REVIEW_PATH = os.path.join(DATA_DIR, "qa_benchmark_600_source_review.csv")
METADATA_PATH = os.path.join(DATA_DIR, "qa_benchmark_600_source_metadata.json")

SOURCES = {
    "JECFA_DB": {
        "title": "WHO/FAO JECFA Food Additives Database",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/",
        "accessed": "2026-10-17",
        "locator": "Database overview and individual chemical evaluation records; use the record-specific page for claims about a substance.",
    },
    "FAO_JECFA_SPECS": {
        "title": "FAO JECFA Specifications for Food Additives",
        "url": "https://www.fao.org/food/food-safety-quality/scientific-advice/jecfa/jecfa-additives/en/",
        "accessed": "2026-10-17",
        "locator": "FAO JECFA specifications page; the current specification is in the monograph with the highest number.",
    },
    "WHO_ASPARTAME_2023": {
        "title": "WHO/FAO/IARC: Aspartame hazard and risk assessment results",
        "url": "https://www.who.int/news/item/14-07-2023-aspartame-hazard-and-risk-assessment-results-released",
        "accessed": "2026-10-17",
        "locator": "WHO/FAO/IARC joint 2023 announcement; distinguishes IARC hazard classification from JECFA risk assessment.",
    },
    "JECFA_ASPARTAME": {
        "title": "WHO/FAO JECFA Aspartame evaluation record",
        "url": "https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/62",
        "accessed": "2026-10-17",
        "locator": "Aspartame chemical evaluation record (chemical ID 62).",
    },
    "EFSA_GLUTAMATES": {
        "title": "EFSA Journal 2017;15(7):4910, re-evaluation of glutamates",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7009848/",
        "accessed": "2026-10-17",
        "locator": "EFSA Journal 2017;15(7):4910; DOI 10.2903/j.efsa.2017.4910; see abstract/conclusions for group ADI.",
    },
    "EFSA_MODIFIED_STARCHES": {
        "title": "EFSA Journal 2017;15(10):4911, re-evaluation of modified starches",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7009865/",
        "accessed": "2026-10-17",
        "locator": "EFSA Journal 2017;15(10):4911; DOI 10.2903/j.efsa.2017.4911; see abstract/conclusions for reported uses and ADI.",
    },
    "EFSA_SULFITES": {
        "title": "EFSA Journal 2022;20(11):7594, follow-up on sulfur dioxide and sulfites",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9685353/",
        "accessed": "2026-10-17",
        "locator": "EFSA Journal 2022;20(11):7594; DOI 10.2903/j.efsa.2022.7594; see conclusions for withdrawal of the temporary group ADI.",
    },
    "CODEX_CXS_1": {
        "title": "FAO-hosted Codex CXS 1-1985 text (front matter: amended through 2010)",
        "url": "https://www.fao.org/input/download/standards/32/CXS_001e.pdf",
        "accessed": "2026-10-17",
        "locator": "Sections 4.2.1.4 (hypersensitivity ingredients), 4.2.2 (allergens transferred through biotechnology), and 6 (limited exemption for small units). The linked PDF contains no Annex II and its front matter lists amendments only through 2010.",
        "scope_warning": "This citation establishes only what is present in the linked text. Verify any later Codex amendment and applicable Vietnamese labelling law before treating the claims as current requirements.",
    },
    "WHO_ANC": {
        "title": "WHO recommendations on antenatal care for a positive pregnancy experience",
        "url": "https://www.who.int/publications/i/item/9789241549912",
        "accessed": "2026-10-17",
        "locator": "WHO antenatal-care guideline; cited only as a general pregnancy-care source, not as an additive-specific intake or safety recommendation.",
    },
    "WHO_HEALTHY_DIET": {
        "title": "WHO Healthy diet fact sheet",
        "url": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
        "accessed": "2026-10-17",
        "locator": "Sections on healthy diet principles, salt, fats, sugars, and healthy diets for infants and young children; age-specific produce/fibre targets are stated in the latter section.",
    },
    "WHO_COMPLEMENTARY_FEEDING": {
        "title": "WHO guideline for complementary feeding of infants and young children 6-23 months",
        "url": "https://www.who.int/publications/i/item/9789240081864",
        "accessed": "2026-10-17",
        "locator": "Overview and recommendations for complementary feeding of children 6-23 months; guideline scope is population-level.",
    },
    "USDA_FDC": {
        "title": "USDA FoodData Central download datasets and FNDDS files",
        "url": "https://fdc.nal.usda.gov/download-datasets/",
        "accessed": "2026-10-17",
        "locator": "USDA FoodData Central download datasets; QA values are read from the local FNDDS food, food_nutrient, and nutrient tables, per 100 g.",
    },
}

QUESTION_TEMPLATES = {
    "additive_safety": [
        "Theo nguồn {source_label}, {question}",
        "Thông tin chính thức về {topic}: {question}",
        "Tra cứu nguồn {source_label}: {question}",
        "Có thể trả lời thế nào dựa riêng trên {source_label}: {question}",
        "Dựa trên bằng chứng được trích dẫn, {question}",
        "Kết luận nào được nguồn cho phép về {topic}? {question}",
        "Trong phạm vi thông tin của {source_label}, {question}",
        "Xin đối chiếu với nguồn {source_label}: {question}",
        "Câu trả lời có căn cứ từ tài liệu {source_label} là gì? {question}",
        "Theo hồ sơ nguồn được cung cấp về {topic}, {question}",
    ],
    "allergen_safety": [
        "Theo Codex CXS 1-1985, {question}",
        "Về ghi nhãn thực phẩm đóng gói, {question}",
        "Tra cứu danh mục dị nguyên Codex: {question}",
        "Codex đề cập gì về {topic}? {question}",
        "Dựa trên tiêu chuẩn được trích dẫn, {question}",
        "Thông tin ghi nhãn nào được Codex nêu cho {topic}? {question}",
        "Theo tiêu chuẩn Codex về thực phẩm đóng gói, {question}",
        "Hãy trả lời đúng phạm vi của CXS 1-1985: {question}",
        "Nguồn Codex có liệt kê {topic} không? {question}",
        "Theo quy tắc thông tin dị nguyên được trích dẫn, {question}",
    ],
    "pregnancy_safety": [
        "Với người đang mang thai, {question}",
        "Có thể kết luận gì về thai kỳ từ nguồn này? {question}",
        "Nguồn được trích dẫn có đủ để khẳng định {question}",
        "Câu trả lời thận trọng, có căn cứ cho thai phụ là gì? {question}",
        "Dựa trên dữ liệu về {topic}, {question}",
        "Thông tin này có phải khuyến nghị riêng cho thai kỳ không? {question}",
        "Đối với một thai phụ cụ thể, {question}",
        "Nguồn đánh giá phụ gia có thay thế tư vấn thai kỳ không? {question}",
        "Điều gì còn chưa được xác định cho trường hợp mang thai? {question}",
        "Hãy phân biệt ADI quần thể với kết luận cá nhân: {question}",
        "Theo nguồn hiện có, có thể xác nhận an toàn trong thai kỳ không? {question}",
        "Nguồn này có thiết lập liều riêng cho thai phụ không? {question}",
        "Có nên dùng ADI như đơn thuốc cho thai phụ? {question}",
        "Thai phụ cần hiểu giới hạn của bằng chứng này thế nào? {question}",
        "Thông tin nào có thể nêu mà không suy đoán cho thai phụ? {question}",
        "Bằng chứng được trích dẫn có trả lời nguy cơ cá nhân trong thai kỳ không? {question}",
        "Có đủ dữ liệu để đưa lời khuyên sử dụng cho thai phụ không? {question}",
        "Khi nguồn chỉ đưa ra ADI, kết luận riêng cho thai kỳ là gì? {question}",
        "Hãy trả lời câu hỏi về thai kỳ mà không vượt quá bằng chứng: {question}",
        "Trường hợp thai phụ có cần được chuyển tư vấn chuyên môn không? {question}",
    ],
    "children_safety": [
        "{question}",
        "Theo WHO, {question}",
        "Trong hướng dẫn WHO, {question}",
        "WHO đưa ra khuyến nghị nào cho câu hỏi: {question}",
        "Đối chiếu với hướng dẫn WHO: {question}",
        "Xin nêu câu trả lời theo khuyến nghị WHO: {question}",
        "Thông tin WHO về câu hỏi này là gì: {question}",
        "Với nhóm tuổi liên quan, {question}",
        "Căn cứ tài liệu WHO, {question}",
        "Hãy tìm câu trả lời trong khuyến nghị WHO: {question}",
    ],
    "condition_safety": [
        "{question}",
        "Theo nguồn WHO: {question}",
        "Dựa trên nguồn WHO, hãy trả lời: {question}",
        "WHO nêu khuyến nghị quần thể nào cho câu hỏi: {question}",
        "Thông tin được trích dẫn có trả lời câu hỏi sau không: {question}",
        "Hãy trả lời trong đúng phạm vi khuyến nghị WHO: {question}",
        "Nguồn WHO cho biết gì về vấn đề sau: {question}",
        "Căn cứ tài liệu WHO: {question}",
        "Câu trả lời nào được hỗ trợ bởi nguồn WHO: {question}",
        "Tra cứu khuyến nghị WHO: {question}",
    ],
}


ADDITIVE_FACTS = [
    {
        "key": "jecfa_scope",
        "topic": "phạm vi JECFA",
        "source_ids": ["JECFA_DB"],
        "question": "JECFA đánh giá những nhóm chất nào theo mô tả cơ sở dữ liệu?",
        "answer": "Cơ sở dữ liệu WHO/FAO JECFA bao gồm các đánh giá về hương liệu, phụ gia thực phẩm, chất ô nhiễm, độc tố và thuốc thú y.",
        "evidence": "The searchable database contains summaries of evaluations of flavours, food additives, contaminants, toxicants and veterinary drugs.",
    },
    {
        "key": "jecfa_record_contents",
        "topic": "nội dung hồ sơ JECFA",
        "source_ids": ["JECFA_DB"],
        "question": "Một bản tóm tắt đánh giá JECFA có thể cung cấp những loại thông tin nào?",
        "answer": "Theo mô tả của WHO/FAO, bản tóm tắt có thông tin hóa học cơ bản, ADI/TDI, liên kết tới báo cáo/monograph và specification mới nhất, cùng lịch sử đánh giá.",
        "evidence": "Each summary contains basic chemical information, ADIs/TDIs, links to the most recent reports and monographs as well as to the specification database, and a history of JECFA evaluations.",
    },
    {
        "key": "jecfa_search",
        "topic": "cách tra cứu JECFA",
        "source_ids": ["JECFA_DB"],
        "question": "Cơ sở dữ liệu WHO/FAO cho phép tra cứu bằng những khóa nào?",
        "answer": "Theo mô tả cơ sở dữ liệu, có thể tra theo một phần tên, số CAS hoặc functional class.",
        "evidence": "The database is searchable by partial name or CAS number, by first character (letter or symbol), or by functional class.",
    },
    {
        "key": "fao_specification_version",
        "topic": "phiên bản specification JECFA",
        "source_ids": ["FAO_JECFA_SPECS"],
        "question": "Theo FAO, monograph nào chứa specification hiện hành cần dùng?",
        "answer": "FAO hướng dẫn dùng specification trong monograph có số cao nhất; specification hiện hành cần được kiểm tra ở monograph mới nhất thay vì dựa vào bản cũ.",
        "evidence": "The current version of the specifications is that included in the Monograph with the highest number.",
    },
    {
        "key": "aspartame_adi",
        "topic": "ADI aspartame",
        "source_ids": ["WHO_ASPARTAME_2023", "JECFA_ASPARTAME"],
        "question": "WHO/FAO JECFA đã duy trì ADI nào cho aspartame vào năm 2023?",
        "answer": "JECFA khẳng định lại ADI 0-40 mg/kg thể trọng/ngày cho aspartame. Đây là ADI, không phải lượng dùng khuyến nghị hay đánh giá liều của một sản phẩm cụ thể.",
        "evidence": "JECFA concluded there was no sufficient reason to change the previously established ADI of 0-40 mg/kg body weight for aspartame.",
    },
    {
        "key": "aspartame_iarc",
        "topic": "phân biệt đánh giá aspartame",
        "source_ids": ["WHO_ASPARTAME_2023"],
        "question": "WHO/IARC đã phân loại aspartame ra sao, và JECFA có thay đổi ADI không?",
        "answer": "IARC xếp aspartame vào nhóm 2B (có thể gây ung thư cho người) dựa trên bằng chứng hạn chế; riêng JECFA kết luận không đủ lý do để thay đổi ADI 0-40 mg/kg thể trọng/ngày.",
        "evidence": "IARC classified aspartame as possibly carcinogenic to humans (Group 2B) on limited evidence; JECFA reaffirmed the ADI of 0-40 mg/kg body weight.",
    },
    {
        "key": "hazard_vs_risk",
        "topic": "hazard và risk",
        "source_ids": ["WHO_ASPARTAME_2023"],
        "question": "Phân loại nguy cơ IARC có cho biết xác suất mắc bệnh ở một mức tiêu thụ cụ thể không?",
        "answer": "Không. WHO giải thích phân loại IARC phản ánh sức mạnh bằng chứng về khả năng một tác nhân có thể gây ung thư, không phản ánh nguy cơ mắc bệnh ở một mức phơi nhiễm cụ thể.",
        "evidence": "IARC classifications reflect the strength of scientific evidence as to whether an agent can cause cancer in humans, but they do not reflect the risk of developing cancer at a given exposure level.",
    },
    {
        "key": "glutamate_adi",
        "topic": "ADI nhóm glutamate",
        "source_ids": ["EFSA_GLUTAMATES"],
        "question": "ADI nhóm của glutamic acid và các glutamate E620-E625 được biểu thị theo chất nào?",
        "answer": "EFSA đặt ADI nhóm 30 mg/kg thể trọng/ngày, được biểu thị theo glutamic acid, cho glutamic acid và các glutamate E620-E625.",
        "evidence": "The Panel derived a group ADI of 30 mg/kg bw per day, expressed as glutamic acid, for glutamic acid and glutamates (E 620-625).",
    },
    {
        "key": "modified_starch_adi",
        "topic": "đánh giá modified starch",
        "source_ids": ["EFSA_MODIFIED_STARCHES"],
        "question": "EFSA kết luận gì về ADI dạng số và mức sử dụng được báo cáo của nhóm modified starches?",
        "answer": "Đánh giá EFSA 2017 bao gồm E1420 trong 12 modified starches; không cần ADI dạng số và không có quan ngại an toàn ở mức sử dụng/mức dùng được báo cáo cho dân số chung. Không suy rộng thành đảm bảo cho mọi liều hoặc mọi cá nhân.",
        "evidence": "The opinion includes acetylated starch (E 1420); no need for a numerical ADI and no safety concern at the reported uses and use levels for the general population.",
    },
    {
        "key": "sulfite_adi_update",
        "topic": "cập nhật đánh giá sulfite",
        "source_ids": ["EFSA_SULFITES"],
        "question": "EFSA 2022 đã cập nhật thế nào ADI tạm thời trước đó của sulfites?",
        "answer": "EFSA kết luận dữ liệu độc tính vẫn chưa đủ để thiết lập ADI và rút ADI nhóm tạm thời 0.7 mg SO2 tương đương/kg thể trọng/ngày; đánh giá tiếp theo dùng phương pháp margin of exposure. Đây là cập nhật EFSA, không phải JECFA.",
        "evidence": "The Panel considered the available toxicity database inadequate to derive an ADI and withdrew the current temporary group ADI; a margin of exposure approach was considered appropriate.",
    },
]

ALLERGENS = [
    ("cereals containing gluten", "ngũ cốc chứa gluten"),
    ("crustaceans", "giáp xác"),
    ("eggs", "trứng"),
    ("fish", "cá"),
    ("peanuts", "lạc/đậu phộng"),
    ("soybeans", "đậu nành"),
    ("milk", "sữa"),
    ("tree nuts", "hạt cây"),
    ("sulphites", "sulfite"),
]
ALLERGEN_SOURCE_DETAILS = {
    "cereals containing gluten": "Mục 4.2.1.4 nêu lúa mì, lúa mạch đen, đại mạch, yến mạch, spelt, các giống lai và sản phẩm từ các loại này.",
    "crustaceans": "Mục 4.2.1.4 nêu giáp xác và sản phẩm từ giáp xác.",
    "eggs": "Mục 4.2.1.4 nêu trứng và sản phẩm từ trứng.",
    "fish": "Mục 4.2.1.4 nêu cá và sản phẩm từ cá.",
    "peanuts": "Mục 4.2.1.4 liệt kê lạc cùng đậu nành và các sản phẩm từ hai nhóm này.",
    "soybeans": "Mục 4.2.1.4 liệt kê lạc cùng đậu nành và các sản phẩm từ hai nhóm này.",
    "milk": "Mục 4.2.1.4 nêu sữa và sản phẩm từ sữa, bao gồm lactose.",
    "tree nuts": "Mục 4.2.1.4 nêu hạt cây và sản phẩm từ hạt.",
    "sulphites": "Mục 4.2.1.4 áp dụng với sulfite ở nồng độ 10 mg/kg trở lên.",
}
ALLERGEN_QUESTION_FORMS = [
    "Codex CXS 1-1985 có nêu {vi} trong các nhóm cần lưu ý khi ghi nhãn dị nguyên không?",
    "Mục 4.2.1.4 của CXS 1-1985 có nêu nhóm {vi} không?",
    "Tiêu chuẩn Codex được trích dẫn đề cập thế nào tới {vi} trong ghi nhãn?",
    "Có thể xác nhận từ CXS 1-1985 rằng {vi} là một nhóm dị nguyên được đề cập không?",
    "Trong quy tắc ghi nhãn thực phẩm đóng gói, {vi} có được tiêu chuẩn này nêu không?",
    "Hãy kiểm tra nhóm {vi} theo danh sách trong CXS 1-1985.",
    "CXS 1-1985 có đề cập nguồn dị nguyên {vi} không?",
    "Về mặt ghi nhãn theo Codex, nhóm {vi} có nằm trong phạm vi tiêu chuẩn được trích dẫn không?",
    "Nguồn Codex này nêu gì về khai báo {vi}?",
    "Câu trả lời có căn cứ nào về {vi} trong CXS 1-1985?",
]
ALLERGEN_ANSWER = (
    "Trong bản CXS 1-1985 được liên kết, mục 4.2.1.4 nêu {vi} trong nhóm thực phẩm/ "
    "thành phần được biết là gây phản ứng quá mẫn và yêu cầu khai báo. {detail} "
    "Mục 6 cho phép "
    "miễn một số yêu cầu mục 4.2 đối với đơn vị nhỏ có diện tích bề mặt lớn nhất dưới "
    "10 cm² (trừ gia vị và thảo mộc); việc miễn áp dụng là tùy quy định. Đây là nội dung "
    "ghi nhãn của bản văn được dẫn, không chẩn đoán dị ứng hay xác định dung nạp cá nhân; "
    "cần xác minh sửa đổi Codex mới hơn và luật Việt Nam trước khi áp dụng."
)
ALLERGEN_EVIDENCE = (
    "FAO-hosted Codex CXS 1-1985 text, section 4.2.1.4: 'The following foods and "
    "ingredients are known to cause hypersensitivity and shall always be declared'; "
    "section 6: small units with largest surface area <10 cm² may be exempted from "
    "paragraph 4.2 requirements, except spices and herbs. Front matter lists amendments "
    "through 2010; verify later amendments and local law."
)

PREGNANCY_CASES = [
    {
        "name": "aspartame",
        "source_ids": ["WHO_ASPARTAME_2023", "JECFA_ASPARTAME", "WHO_ANC"],
        "evidence": "WHO/FAO JECFA reaffirmed aspartame ADI 0-40 mg/kg bw/day; the WHO antenatal-care guideline provides pregnancy-care recommendations but does not establish a product-specific intake from a package label.",
        "answer": "Nguồn WHO/FAO nêu ADI 0-40 mg/kg thể trọng/ngày cho aspartame. Thông tin này không tự xác định lượng aspartame trong sản phẩm hay đưa ra kết luận an toàn cá nhân cho thai kỳ; cần xác minh mức phơi nhiễm và hỏi nhân viên chăm sóc thai sản.",
    },
    {
        "name": "glutamate E620-E625",
        "source_ids": ["EFSA_GLUTAMATES", "WHO_ANC"],
        "evidence": "EFSA group ADI 30 mg/kg bw/day expressed as glutamic acid for E620-E625; WHO antenatal-care guidance is person-centred pregnancy care, not a product-specific additive assessment.",
        "answer": "Nguồn EFSA nêu ADI nhóm 30 mg/kg thể trọng/ngày tính theo glutamic acid cho E620-E625. Dữ liệu này không cho biết lượng trong sản phẩm hoặc quyết định phù hợp cho một thai phụ cụ thể; cần tư vấn nhân viên y tế theo tình trạng và khẩu phần thực tế.",
    },
    {
        "name": "sulfites E220-E228",
        "source_ids": ["EFSA_SULFITES", "WHO_ANC"],
        "evidence": "EFSA 2022 withdrew the temporary group ADI for sulfites and used a margin-of-exposure approach; the assessment is population-level and not a pregnancy-specific product threshold.",
        "answer": "EFSA 2022 rút ADI nhóm tạm thời của sulfites và sử dụng phương pháp margin of exposure; đây không phải ngưỡng cá nhân cho thai kỳ. Không thể kết luận sản phẩm an toàn hay nguy hiểm cho một thai phụ chỉ từ mã phụ gia; cần hỏi nhân viên chăm sóc thai sản.",
    },
    {
        "name": "modified starch E1420",
        "source_ids": ["EFSA_MODIFIED_STARCHES", "WHO_ANC"],
        "evidence": "EFSA 2017 included E1420 in its modified-starch re-evaluation and found no need for a numerical ADI at reported uses/use levels for the general population; this is not an individual pregnancy assessment.",
        "answer": "EFSA không yêu cầu ADI dạng số cho modified starches ở các mức sử dụng được báo cáo và không nêu quan ngại cho dân số chung. Không nên chuyển kết luận quần thể đó thành bảo đảm cá nhân cho thai kỳ hoặc cho mọi lượng dùng; hãy trao đổi với nhân viên chăm sóc thai sản nếu cần tư vấn.",
    },
    {
        "name": "lecithin E322",
        "source_ids": ["CODEX_CXS_1", "WHO_ANC"],
        "evidence": "Codex CXS 1-1985 is a prepackaged-food labelling standard; WHO antenatal care is person-centred. These sources do not establish the amount/source of lecithin in a product or an individual pregnancy risk.",
        "answer": "Các nguồn được trích dẫn không xác định nguồn hoặc lượng lecithin trong sản phẩm cụ thể và không đưa ra kết luận nguy cơ cá nhân trong thai kỳ. Cần đọc nhãn đầy đủ, xác minh thành phần nếu cần và hỏi nhân viên y tế; không thể kết luận chỉ từ mã E322.",
    },
]
PREGNANCY_QUESTIONS = [
    "Nguồn này có đủ để kết luận {name} an toàn cho thai phụ cụ thể không?",
    "ADI/đánh giá được nêu có phải mức dùng được khuyến nghị trong thai kỳ đối với {name} không?",
    "Có thể xác định lượng {name} trong sản phẩm chỉ từ ADI không?",
    "Nguồn WHO antenatal care có đưa ra liều riêng cho {name} không?",
    "Có nên dùng kết luận dân số chung về {name} như bảo đảm cho mọi thai phụ không?",
    "Cần trả lời thế nào nếu thai phụ hỏi sản phẩm có {name} có an toàn không?",
    "Các tài liệu này có xác nhận ngưỡng phơi nhiễm cá nhân khi mang thai cho {name} không?",
    "Có thể khuyên một thai phụ dùng sản phẩm chứa {name} chỉ dựa vào thông tin này không?",
    "Bằng chứng được cung cấp nói gì và chưa nói gì về {name} trong thai kỳ?",
    "Thông tin nào cần bổ sung trước khi đánh giá phơi nhiễm {name} khi mang thai?",
    "Nguồn hiện có có thay thế đánh giá của bác sĩ sản khoa về {name} không?",
    "Nếu chỉ biết mã {name}, có thể kết luận về an toàn thai kỳ không?",
    "Đánh giá ADI có cung cấp hàm lượng {name} trong một khẩu phần cụ thể không?",
    "Tài liệu có đưa ra khuyến nghị riêng theo tam cá nguyệt cho {name} không?",
    "Nên phân biệt đánh giá quần thể và tư vấn cá nhân về {name} thế nào?",
    "Có thể gắn nhãn sản phẩm chứa {name} là an toàn cho thai kỳ từ nguồn này không?",
    "Khi thiếu lượng phơi nhiễm, kết luận nào phù hợp cho câu hỏi về {name}?",
    "Nguồn có chứng minh {name} gây hại cho mọi thai phụ không?",
    "Nguồn có chứng minh {name} hoàn toàn an toàn ở mọi liều trong thai kỳ không?",
    "Hướng xử lý thận trọng cho câu hỏi thai kỳ liên quan {name} là gì?",
]

CHILD_FACTS = [
    ("exclusive_breastfeeding", "WHO khuyến nghị trẻ được bú mẹ hoàn toàn trong 6 tháng đầu đời.", "WHO healthy diet fact sheet states infants should be exclusively breastfed during the first 6 months.", "Trẻ nên được bú mẹ hoàn toàn trong bao lâu?"),
    ("continued_breastfeeding", "WHO khuyến nghị tiếp tục cho trẻ bú mẹ đến 2 tuổi hoặc lâu hơn.", "WHO healthy diet fact sheet states infants should be breastfed continuously until 2 years of age and beyond.", "Đến độ tuổi nào nên tiếp tục cho trẻ bú mẹ?"),
    ("complementary_start", "WHO nêu thức ăn bổ sung bắt đầu khoảng 6 tháng tuổi, khi sữa mẹ hoặc sữa công thức không còn đáp ứng đủ nhu cầu dinh dưỡng.", "WHO complementary feeding guideline defines complementary feeding as foods in addition to milk, generally starting at 6 months.", "Khi nào thường bắt đầu cho trẻ ăn thức ăn bổ sung?"),
    ("no_added_salt_sugar", "WHO khuyến nghị không thêm muối và đường vào thức ăn bổ sung.", "WHO healthy diet fact sheet: salt and sugars should not be added to complementary foods.", "Có nên thêm muối hoặc đường vào thức ăn bổ sung cho trẻ?"),
    ("child_diverse_diet", "Đối với trẻ 6-23 tháng, WHO nêu chế độ ăn đa dạng gồm thực phẩm nguồn động vật (như thịt, cá hoặc trứng), rau quả.", "WHO healthy diet fact sheet: children 6-23 months should consume a diverse diet including animal-source foods and fruits and vegetables.", "Chế độ ăn đa dạng cho trẻ 6-23 tháng nên gồm những nhóm thực phẩm nào?"),
    ("avoid_sweet_drinks", "WHO khuyến nghị tránh đồ uống ngọt và thực phẩm nhiều đường, muối hoặc trans fat đối với trẻ nhỏ.", "WHO healthy diet fact sheet lists foods high in sugar, salt and trans fats as well as sweet beverages to be avoided for infants and young children.", "Những loại thực phẩm hoặc đồ uống nào nên tránh cho trẻ nhỏ theo WHO?"),
    ("fruit_veg_2_5", "WHO nêu trẻ 2-5 tuổi nên hướng tới ít nhất 250 g rau quả mỗi ngày.", "WHO healthy diet fact sheet recommends at least 250 g fruits and vegetables for children aged 2-5 years.", "Trẻ 2-5 tuổi nên ăn ít nhất bao nhiêu rau quả mỗi ngày?"),
    ("fruit_veg_6_9", "WHO nêu trẻ 6-9 tuổi nên hướng tới ít nhất 350 g rau quả mỗi ngày.", "WHO healthy diet fact sheet recommends at least 350 g fruits and vegetables for children aged 6-9 years.", "Trẻ 6-9 tuổi nên ăn ít nhất bao nhiêu rau quả mỗi ngày?"),
    ("fruit_veg_over10", "WHO nêu người trên 10 tuổi nên hướng tới ít nhất 400 g rau quả mỗi ngày.", "WHO healthy diet fact sheet recommends at least 400 g fruits and vegetables for everyone older than 10 years.", "Người trên 10 tuổi nên ăn ít nhất bao nhiêu rau quả mỗi ngày?"),
    ("fiber_children", "WHO nêu lượng chất xơ tự nhiên tối thiểu: 15 g/ngày ở trẻ 2-5 tuổi, 21 g/ngày ở trẻ 6-9 tuổi và 25 g/ngày ở người trên 10 tuổi.", "WHO healthy diet fact sheet gives naturally occurring dietary fibre targets: at least 15 g/day (ages 2-5), 21 g/day (ages 6-9), and 25 g/day (>10 years).", "Lượng chất xơ tự nhiên tối thiểu được nêu cho từng nhóm tuổi là bao nhiêu?"),
]
CHILD_QUESTION_FORMS = [
    "{question}",
    "Theo WHO, hãy xác định: {question}",
    "Trong hướng dẫn WHO, câu trả lời cho câu hỏi sau là gì: {question}",
    "WHO đưa ra thông tin nào để trả lời: {question}",
    "Đối chiếu với hướng dẫn WHO: {question}",
    "Xin nêu câu trả lời theo khuyến nghị WHO cho câu hỏi: {question}",
    "Thông tin WHO về vấn đề này là gì: {question}",
    "Với nhóm tuổi liên quan, hãy trả lời: {question}",
    "Căn cứ tài liệu WHO, câu trả lời là gì: {question}",
    "Hãy tìm câu trả lời trong khuyến nghị WHO cho câu hỏi: {question}",
]

CONDITION_FACTS = [
    ("diet_principles", "Bốn nguyên tắc nền tảng của chế độ ăn lành mạnh theo WHO là đầy đủ, cân bằng, điều độ và đa dạng.", "WHO healthy diet fact sheet names adequacy, balance, moderation and diversity as four core principles.", "Bốn nguyên tắc nền tảng của chế độ ăn lành mạnh là gì?"),
    ("diet_varies", "WHO nêu thành phần chính xác của chế độ ăn lành mạnh thay đổi theo đặc điểm cá nhân, bối cảnh văn hóa, thực phẩm sẵn có và tập quán.", "WHO healthy diet fact sheet says the exact make-up varies by individual characteristics, cultural context, locally available foods and dietary customs.", "Những yếu tố nào làm thành phần chế độ ăn lành mạnh thay đổi giữa các cá nhân?"),
    ("carbohydrate_range", "WHO nêu carbohydrate có thể cung cấp khoảng 45-75% tổng năng lượng hằng ngày cho đa số người; chế độ ăn thấp carb có thể dùng để điều trị một số tình trạng.", "WHO healthy diet fact sheet gives approximately 45-75% total daily energy from carbohydrates for most people and notes low/very-low-carb diets are used to treat certain conditions.", "Khoảng năng lượng từ carbohydrate nào được nêu cho đa số người?"),
    ("free_sugars_10", "WHO khuyến nghị giới hạn đường tự do dưới 10% tổng năng lượng hằng ngày.", "WHO healthy diet fact sheet: free sugars should be limited to less than 10% of total daily energy intake.", "Đường tự do nên chiếm dưới bao nhiêu phần trăm tổng năng lượng hằng ngày?"),
    ("free_sugars_5", "WHO cho biết giảm đường tự do xuống 5% năng lượng hoặc thấp hơn có thể đem lại lợi ích sức khỏe bổ sung.", "WHO healthy diet fact sheet: limiting free sugars further to 5% or less may provide additional health benefits.", "Mức đường tự do nào có thể đem lại thêm lợi ích sức khỏe?"),
    ("adult_total_fat", "WHO nêu giới hạn tổng chất béo 30% năng lượng trở xuống có thể giúp phòng tăng cân không lành mạnh ở người lớn.", "WHO healthy diet fact sheet: limiting total fat to 30% or less of total energy may help prevent unhealthy weight gain in adults.", "Giới hạn tổng chất béo nào có thể giúp phòng tăng cân không lành mạnh ở người lớn?"),
    ("saturated_fat", "WHO khuyến nghị chất béo bão hòa không quá 10% tổng năng lượng.", "WHO healthy diet fact sheet: no more than 10% total energy should come from saturated fat.", "Tỷ lệ năng lượng tối đa từ chất béo bão hòa theo WHO là bao nhiêu?"),
    ("trans_fat", "WHO khuyến nghị chất béo trans không quá 1% tổng năng lượng; trans fat công nghiệp không thuộc chế độ ăn lành mạnh và nên tránh.", "WHO healthy diet fact sheet: no more than 1% energy from trans fat; industrially produced trans fats should be avoided.", "Giới hạn nào được nêu cho trans fat và nên tránh loại trans fat nào?"),
    ("adult_salt", "WHO khuyến nghị người lớn tiêu thụ dưới 5 g muối mỗi ngày, tương đương 2 g natri.", "WHO healthy diet fact sheet: adults should limit salt to less than 5 g/day (2 g/day sodium).", "Giới hạn muối và natri hằng ngày cho người lớn là bao nhiêu?"),
    ("lower_sodium_salt", "WHO lưu ý muối thay thế giảm natri có kali không phù hợp với người có nguy cơ tăng kali máu và cần năng lực phát hiện/quản lý bệnh thận.", "WHO healthy diet fact sheet qualifies lower-sodium salt substitutes: not for people at risk of hyperkalaemia; recommends in settings able to detect/manage kidney disease.", "Khuyến nghị về muối thay thế giảm natri có kali lưu ý gì với người có nguy cơ tăng kali máu?"),
]

NUTRIENT_FIELDS = [
    ("calories_kcal", "năng lượng", "kcal/100 g"),
    ("protein_g", "protein", "g/100 g"),
    ("sodium_mg", "natri", "mg/100 g"),
    ("sugars_g", "đường tổng", "g/100 g"),
    ("fiber_g", "chất xơ", "g/100 g"),
]


def _make_record(record_id, domain, group_id, question, answer, source_ids, evidence,
                 source_refs=None, extra=None):
    source_ids = list(source_ids)
    citations = " ".join(f"[{source_id}]" for source_id in source_ids)
    record = {
        "id": record_id,
        "domain": domain,
        "question": question,
        "reference_answer": f"{answer} {citations}",
        "source_ids": source_ids,
        "source_refs": source_refs or [],
        "source_urls": [SOURCES[source_id]["url"] for source_id in source_ids],
        "evidence_excerpt": evidence,
        "question_group_id": group_id,
        "split_group_id": group_id,
        "generation_method": "source_fact_with_controlled_question_variants",
        "review_status": "draft_requires_source_and_expert_review",
        "human_review_required": True,
        "is_expert_validated_gold": False,
        "not_individualized_medical_advice": True,
    }
    if extra:
        record.update(extra)
    return record


def _add_fact_variants(records, domain, fact, templates):
    count = len(templates)
    for variant, template in enumerate(templates):
        question = template.format(
            source_label=", ".join(SOURCES[sid]["title"] for sid in fact["source_ids"]),
            topic=fact["topic"],
            question=fact["question"],
        )
        records.append(_make_record(
            record_id=f"SGQA-{len(records) + 1:04d}",
            domain=domain,
            group_id=f"{domain}:{fact['key']}",
            question=question,
            answer=fact["answer"],
            source_ids=fact["source_ids"],
            evidence=fact["evidence"],
            source_refs=fact.get("source_refs"),
            extra={
                "answer_type": "source_fact",
                "question_variant": variant + 1,
                "variants_in_group": count,
            },
        ))


def _build_nutrition_records(records):
    foods = load_fndds_nutrition(DATA_DIR)
    candidates = sorted(
        foods,
        key=lambda item: (
            item.get("food_category", ""),
            item.get("food_code", ""),
        ),
    )
    selected = []
    seen_categories = set()
    for food in candidates:
        category = food.get("food_category", "")
        if not category or category in seen_categories:
            continue
        if not food.get("food_code") or not food.get("food_description"):
            continue
        if not all(
            isinstance(food.get(field), (int, float))
            and math.isfinite(food[field])
            and food[field] > 0
            for field, _, _ in NUTRIENT_FIELDS
        ):
            continue
        selected.append(food)
        seen_categories.add(category)
        if len(selected) == 20:
            break
    if len(selected) < 20:
        raise ValueError(f"Need 20 distinct FNDDS categories; found {len(selected)}")

    for food in selected:
        code = str(food["food_code"])
        for field, label, unit in NUTRIENT_FIELDS:
            value = food[field]
            answer = (
                f"USDA FNDDS ghi nhận {value:g} {unit} cho mục "
                f"“{food['food_description']}” (FDC ID {code}). Đây là giá trị "
                "cho mục thực phẩm FNDDS tương ứng, không phải số liệu đã xác nhận "
                "cho mọi sản phẩm cùng tên hoặc một khẩu phần thực tế."
            )
            question = (
                f"Theo USDA FNDDS, mục “{food['food_description']}” "
                f"(FDC ID {code}) có bao nhiêu {label} trên 100 g?"
            )
            records.append(_make_record(
                record_id=f"SGQA-{len(records) + 1:04d}",
                domain="nutrition_facts",
                group_id=f"nutrition:{code}:{field}",
                question=question,
                answer=answer,
                source_ids=["USDA_FDC"],
                evidence=(
                    f"food.fdc_id={code}; food.description={food['food_description']}; "
                    f"{field}={value:g}; nutrient basis=per 100 g from the local "
                    "FNDDS food/food_nutrient/nutrient tables."
                ),
                source_refs=["data/food.csv", "data/food_nutrient.csv", "data/nutrient.csv"],
                extra={
                    "answer_type": "dataset_lookup",
                    "food_code": code,
                    "nutrient_field": field,
                    "value_per_100g": value,
                },
            ))


def build_records():
    records = []
    for fact in ADDITIVE_FACTS:
        _add_fact_variants(records, "additive_safety", fact, QUESTION_TEMPLATES["additive_safety"])

    for allergen_index, (english_name, vietnamese_name) in enumerate(ALLERGENS):
        fact_key = f"codex_allergen_{allergen_index + 1:02d}"
        for variant, template in enumerate(ALLERGEN_QUESTION_FORMS):
            question = template.format(en=english_name, vi=vietnamese_name)
            source_detail = ALLERGEN_SOURCE_DETAILS[english_name]
            answer = ALLERGEN_ANSWER.format(
                vi=vietnamese_name,
                detail=source_detail,
            )
            records.append(_make_record(
                record_id=f"SGQA-{len(records) + 1:04d}",
                domain="allergen_safety",
                group_id=f"allergen:{fact_key}",
                question=question,
                answer=answer,
                source_ids=["CODEX_CXS_1"],
                evidence=f"{ALLERGEN_EVIDENCE} {source_detail}",
                extra={
                    "answer_type": "labeling_standard_lookup",
                    "question_variant": variant + 1,
                    "variants_in_group": len(ALLERGEN_QUESTION_FORMS),
                },
            ))
    standard_boundary = {
        "key": "codex_scope",
        "topic": "phạm vi Codex",
        "source_ids": ["CODEX_CXS_1"],
        "question": "CXS 1-1985 có phải là tiêu chuẩn chẩn đoán dị ứng hoặc xác định mức dung nạp cá nhân không?",
        "answer": "Không. CXS 1-1985 là tiêu chuẩn ghi nhãn thực phẩm đóng gói; không phải công cụ chẩn đoán dị ứng, không xác định mức dung nạp cá nhân và không chứng minh sản phẩm không có nguy cơ nhiễm chéo.",
        "evidence": "The title and scope identify CXS 1-1985 as the General Standard for the Labelling of Prepackaged Foods; it is not a clinical diagnostic guideline.",
    }
    _add_fact_variants(
        records,
        "allergen_safety",
        standard_boundary,
        QUESTION_TEMPLATES["allergen_safety"],
    )

    for fact in PREGNANCY_CASES:
        source_label = ", ".join(SOURCES[sid]["title"] for sid in fact["source_ids"])
        for variant, template in enumerate(PREGNANCY_QUESTIONS):
            question = template.format(name=fact["name"])
            records.append(_make_record(
                record_id=f"SGQA-{len(records) + 1:04d}",
                domain="pregnancy_safety",
                group_id=f"pregnancy:{fact['name']}",
                question=f"{question} (Nguồn: {source_label}.)",
                answer=fact["answer"],
                source_ids=fact["source_ids"],
                evidence=fact["evidence"],
                extra={
                    "answer_type": "evidence_boundary_and_abstention",
                    "question_variant": variant + 1,
                    "variants_in_group": len(PREGNANCY_QUESTIONS),
                    "requires_clinician_referral": True,
                },
            ))

    for key, answer, evidence, question_seed in CHILD_FACTS:
        for variant, template in enumerate(CHILD_QUESTION_FORMS):
            question = template.format(
                question=question_seed,
                topic=key.replace("_", " "),
            )
            records.append(_make_record(
                record_id=f"SGQA-{len(records) + 1:04d}",
                domain="children_safety",
                group_id=f"children:{key}",
                question=question,
                answer=answer + " Đây là hướng dẫn dân số; trường hợp sức khỏe cá nhân cần chuyên gia đánh giá.",
                source_ids=(
                    ["WHO_COMPLEMENTARY_FEEDING"]
                    if key == "complementary_start"
                    else ["WHO_HEALTHY_DIET"]
                ),
                evidence=evidence,
                extra={
                    "answer_type": "population_guideline",
                    "question_variant": variant + 1,
                    "variants_in_group": len(CHILD_QUESTION_FORMS),
                },
            ))

    for key, answer, evidence, question_seed in CONDITION_FACTS:
        fact = {
            "key": key,
            "topic": key.replace("_", " "),
            "source_ids": ["WHO_HEALTHY_DIET"],
            "question": question_seed,
            "answer": answer + " Đây là khuyến nghị dân số, không thay thế kế hoạch điều trị hoặc tư vấn dinh dưỡng cá nhân.",
            "evidence": evidence,
        }
        _add_fact_variants(records, "condition_safety", fact, QUESTION_TEMPLATES["condition_safety"])

    _build_nutrition_records(records)
    return records


def write_outputs(records):
    expected = {
        "additive_safety": 100,
        "allergen_safety": 100,
        "pregnancy_safety": 100,
        "children_safety": 100,
        "condition_safety": 100,
        "nutrition_facts": 100,
    }
    counts = {}
    for record in records:
        counts[record["domain"]] = counts.get(record["domain"], 0) + 1
    if counts != expected:
        raise ValueError(f"Unexpected domain counts: {counts}; expected {expected}")
    questions = [record["question"] for record in records]
    if len(set(questions)) != len(questions):
        raise ValueError("Generated question text is not unique")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    review_fields = [
        "id", "domain", "question", "draft_answer", "source_ids_json",
        "source_urls_json", "evidence_excerpt", "question_group_id",
        "review_status", "reviewer_id", "reviewer_qualification",
        "answer_supported_by_source", "citation_correct", "safety_label_correct",
        "final_answer", "review_notes", "review_date",
    ]
    with open(REVIEW_PATH, "w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=review_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "id": record["id"],
                "domain": record["domain"],
                "question": record["question"],
                "draft_answer": record["reference_answer"],
                "source_ids_json": json.dumps(record["source_ids"], ensure_ascii=False),
                "source_urls_json": json.dumps(record["source_urls"], ensure_ascii=False),
                "evidence_excerpt": record["evidence_excerpt"],
                "question_group_id": record["question_group_id"],
                "review_status": "pending",
            })

    metadata = {
        "title": "FoodHealthQA 600 source-grounded draft",
        "record_count": len(records),
        "domain_counts": counts,
        "is_expert_validated_gold": False,
        "human_review_required": True,
        "generation_scope": (
            "600 source-linked questions, 100 per domain. Health claims use official WHO/FAO, "
            "Codex or EFSA evidence; nutrition lookups use USDA FNDDS. Controlled variants "
            "share question_group_id and must remain in one split."
        ),
        "known_limitations": [
            "Controlled question variants are not 600 independent facts; split and score by question_group_id.",
            "Pregnancy and disease questions deliberately abstain from individual safety conclusions where sources do not support them.",
            "Condition-domain facts are general WHO healthy-diet guidance, not condition-specific treatment protocols.",
            "The linked FAO Codex PDF lists amendments only through 2010; allergens are in section 4.2.1.4, not Annex II, and section 6 includes a limited possible small-unit exemption. Verify later amendments and Vietnamese law.",
            "Codex allergen content is a labeling-standard lookup, not clinical allergy advice.",
            "USDA FNDDS values describe the matched database food record, not an unverified branded product.",
            "A qualified reviewer must verify every answer and citation before calling this a gold benchmark.",
        ],
        "sources": SOURCES,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as output:
        json.dump(metadata, output, ensure_ascii=False, indent=2)
        output.write("\n")


def main():
    records = build_records()
    write_outputs(records)
    print(f"Wrote {len(records)} source-grounded draft QA records.")
    print(f"JSONL: {OUTPUT_PATH}")
    print(f"Review CSV: {REVIEW_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
