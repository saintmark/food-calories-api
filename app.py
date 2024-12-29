from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import logging
import json
from google.cloud import vision
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
import backoff
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Google Cloud 配置
GOOGLE_CREDENTIALS_JSON = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
if not GOOGLE_CREDENTIALS_JSON:
    raise ValueError("未找到 Google Cloud 凭证")

# 从环境变量创建凭证
credentials_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
credentials = service_account.Credentials.from_service_account_info(credentials_dict)

# 初始化 Vision API 和 Translate API 客户端
vision_client = vision.ImageAnnotatorClient(credentials=credentials)
translate_client = translate.Client(credentials=credentials)

# 食物重量参考表（克）
FOOD_WEIGHT_REFERENCE = {
    '主食类': {
        'default': 250,
        'keywords': ['饭', '面', '粥', '馒头', '包子'],
        'range': (200, 400)
    },
    '肉类': {
        'default': 200,
        'keywords': ['肉', '鸡', '鸭', '鱼', '牛', '猪'],
        'range': (100, 300)
    },
    '蔬菜类': {
        'default': 150,
        'keywords': ['菜', '青菜', '生菜', '白菜', '萝卜'],
        'range': (100, 200)
    },
    '水果类': {
        'default': 200,
        'keywords': ['果', '苹果', '香蕉', '橙子'],
        'range': (100, 300)
    }
}

# 食物卡路里参考表（每100克）
FOOD_CALORIES_REFERENCE = {
    '主食类': {
        'default': 116,
        'keywords': ['饭', '面', '粥', '馒头', '包子'],
        'range': (100, 150)
    },
    '肉类': {
        'default': 200,
        'keywords': ['肉', '鸡', '鸭', '鱼', '牛', '猪'],
        'range': (150, 250)
    },
    '蔬菜类': {
        'default': 30,
        'keywords': ['菜', '青菜', '生菜', '白菜', '萝卜'],
        'range': (15, 50)
    },
    '水果类': {
        'default': 56,
        'keywords': ['果', '苹果', '香蕉', '橙子'],
        'range': (40, 80)
    }
}

def get_food_category(food_name):
    """根据食物名称判断类别"""
    for category, info in FOOD_WEIGHT_REFERENCE.items():
        if any(keyword in food_name for keyword in info['keywords']):
            return category
    return '主食类'  # 默认归类为主食

def estimate_food_weight(food_name):
    """估算食物重量"""
    category = get_food_category(food_name)
    return FOOD_WEIGHT_REFERENCE[category]['default']

def calculate_calories(food_name, weight):
    """计算食物卡路里"""
    category = get_food_category(food_name)
    calories_per_100g = FOOD_CALORIES_REFERENCE[category]['default']
    return int(calories_per_100g * (weight / 100))

@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=3,
    max_time=30
)
def detect_food(image_content):
    """使用 Google Cloud Vision API 识别食物"""
    try:
        image = vision.Image(content=image_content)
        
        # 调用 Vision API 的标签检测
        label_response = vision_client.label_detection(image=image)
        labels = label_response.label_annotations
        
        # 调用 Vision API 的对象检测
        object_response = vision_client.object_localization(image=image)
        objects = object_response.localized_object_annotations
        
        # 处理识别结果
        food_related_items = []
        
        # 从标签中提取食物相关词
        for label in labels:
            description = label.description.lower()
            if any(keyword in description for keyword in ['food', 'dish', 'cuisine', 'meal', 'fruit', 'vegetable', 'meat']):
                food_related_items.append({
                    'name': label.description,
                    'score': label.score
                })
        
        # 从对象中提取食物相关词
        for obj in objects:
            name = obj.name.lower()
            if any(keyword in name for keyword in ['food', 'dish', 'cuisine', 'meal', 'fruit', 'vegetable', 'meat']):
                food_related_items.append({
                    'name': obj.name,
                    'score': obj.score
                })
        
        if not food_related_items:
            raise ValueError("未能识别出食物")
            
        # 按置信度排序并获取最可能的结果
        food_related_items.sort(key=lambda x: x['score'], reverse=True)
        english_name = food_related_items[0]['name']
        
        # 使用 Google Translate 进行翻译
        translation = translate_client.translate(
            english_name,
            target_language='zh-CN',
            source_language='en'
        )
        
        return translation['translatedText']
            
    except Exception as e:
        logger.error(f"食物识别错误: {str(e)}")
        raise

@app.route('/identify', methods=['POST'])
def identify_food():
    logger.info("收到识别请求")
    
    if 'food_image' not in request.files:
        logger.error("没有文件")
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['food_image']
    if file.filename == '':
        logger.error("文件名为空")
        return jsonify({'error': '没有选择文件'}), 400
    
    try:
        image_content = file.read()
        food_name = detect_food(image_content)
        logger.info(f"识别成功: 食物名称={food_name}")
        
        weight = estimate_food_weight(food_name)
        
        return jsonify({
            'name': food_name,
            'confidence': 0.9,
            'weight': weight
        })
        
    except Exception as e:
        logger.error(f"识别食物时发生错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/calories', methods=['GET'])
def get_calories():
    try:
        food_name = request.args.get('foodName')
        weight = float(request.args.get('weight', 0))
        
        logger.info(f"收到计算卡路里请求: 食物={food_name}, 重量={weight}克")
        
        if not food_name or weight <= 0:
            return jsonify({'error': '参数不完整', 'calories': 0}), 400
        
        calories = calculate_calories(food_name, weight)
        return jsonify({'calories': calories})
        
    except Exception as e:
        logger.error(f"计算卡路里时发生错误: {str(e)}")
        return jsonify({
            'error': str(e),
            'calories': 0
        }), 500

if __name__ == '__main__':
    port = os.getenv('PORT', 5000)
    app.run(host='0.0.0.0', port=port)