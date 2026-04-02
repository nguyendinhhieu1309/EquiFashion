import os

model_root = os.path.join(os.path.dirname(__file__), "checkpoints")

dataset_root = "dataset/EquiFashion_DB/"

openpose_body_model_path = os.path.join(model_root, "body_pose_model.pth")
openpose_hand_model_path = os.path.join(model_root, "hand_pose_model.pth")

sam_model_path = os.path.join(model_root, "sam_vit_h_4b8939.pth")

model_yaml = os.path.join(os.path.dirname(__file__), "configs/cldm_v2.yaml")
my_model_path = os.path.join(model_root, "EqF_100epochs.ckpt")


device = "cuda:0"

# Vietnamese-English dictionary

category_dict = {
    "Đầm": "Dress",
    "Áo blouse": "Blouse",
    "Áo len": "Sweater",
    "Áo khoác": "Coat",
    "Jumpsuit": "Jumpsuit",
    "Quần": "Pant",
    "Váy": "Skirt"
}


style_dict = {
    "Thiết kế nguyên bản": "Original Design",
    "Phong cách đường phố": "Street Style",
    "Phong cách công sở": "Commute",
    "Phong cách Hàn Quốc": "Korean Style",
    "Thanh lịch": "Elegant",
    "Dễ thương": "Sweet",
    "Phong cách phương Tây": "Western Style",
    "Phong cách Nhật Bản": "Japanese Style",
    "Phong cách Anh": "British Style",
    "Vintage": "Vintage",
    "Nghệ thuật": "Artsy",
    "Phong cách hoàng gia": "Courtly Style",
    "Tối giản": "Simple Style",
    "Phong cách nông thôn": "Rural Style",
    "Phong cách học đường": "Campus Style",
    "Quý cô công sở": "Office Lady",
    "Trang phục ở nhà": "Homewear",
    "Thể thao": "Sport",
    "Thường ngày": "Casual",
    "Cao quý": "Noble",
    "Trẻ trung / Xu hướng": "Youth/Pop",
    "Doanh nhân lịch lãm": "Business Gentleman",
    "Tươi mới": "Fresh",
    "Thời thượng": "Trendy",
    "Phong cách Trung Hoa": "Chinese Style",
    "Punk": "Punk",
    "Hip-hop": "Hip-hop",
    "Quyến rũ": "Sexy",
    "Rock": "Rock",
    "Workwear": "Workwear",
    "Công sở": "Office"
}


occasion_dict = {
    "Trường học": "Campus",
    "Ở nhà": "Home",
    "Hẹn hò": "Date",
    "Tiệc tùng": "Party",
    "Công sở": "Office",
    "Thể thao": "Sport",
    "Du lịch": "Travel",
    "Đám cưới": "Wedding",
    "Kinh doanh": "Business"
}


effect_dict = {
    "Tạo dáng thon gọn": "Slimming",
    "Trông trẻ hơn": "Youthful",
    "Tạo cảm giác cao": "Tall",
    "Tôn vòng hông": "Highlight Hips",
    "Làm sáng da": "Brighten Skin",
    "Làm mặt thon": "Face Slimming",
    "Tạo cảm giác cổ dài": "Elongate Neck",
    "Tôn vòng ngực": "Enhance Bust",
    "Tạo vẻ nam tính cơ bắp": "Muscular Look"
}


feeling_dict = {
    "Cảm giác đường cong": "Sense of Curve",
    "Cảm giác nhẹ nhàng bay bổng": "Sense of Agility and Elegance",
    "Cảm giác bó buộc": "Sense of Restraint",
    "Cảm giác nhiều lớp": "Three-dimensional Layering",
    "Cảm giác mờ ảo": "Hazy Sensation",
    "Cảm giác rũ": "Drape Feeling",
    "Cảm giác nặng nề": "Dullness Sensation",
    "Cảm giác tinh nghịch": "Playful Feeling",
    "Cảm giác trẻ trung": "Youthful Feeling",
    "Cảm giác thú vị": "Sense of Fun",
    "Cảm giác thoải mái": "Casual and Relaxed Feeling",
    "Cảm giác sang trọng": "Sense of Atmosphere",
    "Cảm giác đường nét": "Line Feeling",
    "Cảm giác xếp lớp": "Stacking Feeling",
    "Cảm giác trưởng thành": "Mature Feeling",
    "Cảm giác trẻ thơ": "Childlike Feeling",
    "Cảm giác cồng kềnh": "Bulky Feeling",
    "Cảm giác cứng cáp": "Crisp Feeling",
    "Cảm giác nặng": "Heavy Feeling",
    "Cảm giác rừng rậm": "Jungle Feeling",
    "Cảm giác công nghiệp nặng": "Heavy Industry Feeling"
}


attribute_dict = {
    "Độ dài trang phục": "A1",
    "Độ dài tay áo": "A2",
    "Kiểu tay áo": "A3",
    "Kiểu cổ áo": "A4",
    "Kiểu lai áo": "A5"
}


clothing_length_dict = {
    "Cực ngắn": "Ultra-short",
    "Ngắn": "Short",
    "Dài ngang gối": "Knee-length",
    "Dài trung bình": "Mid-length",
    "Dài": "Long"
}


sleeve_length_dict = {
    "Không tay": "Sleeveless",
    "Tay ngắn": "Short Sleeve",
    "Tay lửng": "Elbow-length Sleeve",
    "Tay dài vừa": "Mid-length Sleeve",
    "Tay dài": "Long Sleeve"
}


sleeve_type_dict = {
    "Tay dơi": "Dolman Sleeve",
    "Tay phồng": "Puffed Sleeve",
    "Tay lồng đèn": "Lantern Sleeve",
    "Tay loe": "Flare Sleeve",
    "Tay raglan": "Raglan Sleeve",
    "Tay bèo": "Ruffle Sleeve",
    "Tay quấn": "Wrapped Sleeve",
    "Tay vai chéo": "Raglan Sleeve",
    "Tay cánh tiên": "Flutter Sleeve",
    "Tay công chúa": "Princess Sleeve",
    "Tay xếp lớp": "Layered Sleeve",
    "Tay áo sơ mi": "Shirt Sleeve",
    "Tay cánh hoa": "Petal Sleeve",
    "Tay kimono": "Kimono Sleeve",
    "Tay tiêu chuẩn": "Regular Sleeve",
    "Tay trễ vai": "Drop-shoulder Sleeve",
}


collar_type_dict = {
    "Cổ tròn": "Round Collar",
    "Cổ chữ V": "V-Neck",
    "Cổ vuông": "Square Collar",
    "Cổ vest": "Tailor Collar",
    "Cổ bẻ": "Lapel Collar",
    "Cổ đứng": "Stand Collar",
    "Cổ chữ T": "T-neck",
    "Cổ thuyền": "Boat Neck",
    "Cổ chữ U": "U-Neck",
    "Cổ chữ A": "A-Line Collar",
    "Cổ rũ": "Swinging Collar",
    "Cổ bất đối xứng": "Irregular Collar",
    "Cổ kín": "Closed Collar"
}


hem_dict = {
    "Lai thẳng": "Flat Hem",
    "Lai cong": "Curved Hem",
    "Lai bèo": "Ruffle Hem",
    "Lai nhiều lớp": "Layered Hem",
    "Lai cạp thấp": "Low Waist Hem",
    "Lai sọc": "Striped Hem",
    "Lai gợn sóng": "Wavy Hem",
    "Lai xẻ": "Slit Hem",
    "Lai rũ": "Draped Hem",
    "Lai bất đối xứng": "Irregular Hem",
    "Lai cuộn": "Curled Hem",
    "Lai tua": "Raw Hem",
    "Lai buộc cổ chân": "Ankle-tied Hem",
    "Lai loe": "Flared Hem",
    "Lai viền": "Flanging Hem",
    "Lai rộng": "Loose Hem",
    "Lai ren": "Lace Hem",
    "Lai thắt dây": "Tight-strap Hem",
    "Lai dây rút": "Drawstring Hem",
    "Lai xếp ly": "Pleated Hem"
}
