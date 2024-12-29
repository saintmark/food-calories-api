from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import logging
import json
from google.cloud import vision
from google.oauth2 import service_account
import backoff
from openai import OpenAI
import httpx
from dotenv import load_dotenv

load_dotenv()

# 在应用开始时添加详细的环境变量检查
def check_environment():
    """检查并打印环境变量信息"""
    logger.info("=== 环境变量检查开始 ===")
    
    # 打印所有环境变量（仅打印名称，不打印值）
    all_env_vars = list(os.environ.keys())
    logger.info(f"系统中的所有环境变量名称: {all_env_vars}")
    
    # 特别检查 OPENAI_API_KEY
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        logger.info("OPENAI_API_KEY 已找到，长度为: " + str(len(api_key)))
        logger.info("OPENAI_API_KEY 前10个字符: " + api_key[:10] + "...")
    else:
        logger.error("OPENAI_API_KEY 未找到!")
        
    logger.info("=== 环境变量检查结束 ===")
    return api_key

# 应用初始化
app = Flask(__name__)
CORS(app)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 检查环境变量
OPENAI_API_KEY = check_environment()
if not OPENAI_API_KEY:
    raise ValueError("未找到OPENAI_API_KEY环境变量，请确保在Railway中正确设置了环境变量")




# OpenAI配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("未找到OPENAI_API_KEY环境变量")
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
OPENAI_TIMEOUT = int(os.getenv('OPENAI_TIMEOUT', '30'))

logger.info("正在检查环境变量...")
logger.info(f"所有环境变量: {list(os.environ.keys())}")  # 打印所有可用的环境变量名

logger.info(f"OPENAI_API_KEY 是否存在: {OPENAI_API_KEY is not None}")
if not OPENAI_API_KEY:
    # 尝试其他可能的变量名
    alternative_names = ['OpenAI_API_Key', 'OPENAIAPI_KEY', 'OPENAI_APIKEY']
    for name in alternative_names:
        OPENAI_API_KEY = os.getenv(name)
        if OPENAI_API_KEY:
            logger.info(f"找到替代变量名: {name}")
            break

    if not OPENAI_API_KEY:
        error_msg = "未找到OPENAI_API_KEY环境变量"
        logger.error(error_msg)
        raise ValueError(error_msg)

# 初始化OpenAI客户端配置
try:
    openai_config = {
        'api_key': OPENAI_API_KEY,
        'timeout': httpx.Timeout(OPENAI_TIMEOUT)
    }
    if OPENAI_BASE_URL:
        openai_config['base_url'] = OPENAI_BASE_URL
        
    logger.info("正在初始化OpenAI客户端...")
    openai_client = OpenAI(**openai_config)
    logger.info("OpenAI客户端初始化成功")
except Exception as e:
    logger.error(f"初始化OpenAI客户端时发生错误: {str(e)}")
    raise

openai_client = OpenAI(**openai_config)

# Google Cloud 配置
GOOGLE_CREDENTIALS_JSON = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
if not GOOGLE_CREDENTIALS_JSON:
    raise ValueError("未找到 Google Cloud 凭证")

# 从环境变量创建凭证
credentials_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
credentials = service_account.Credentials.from_service_account_info(credentials_dict)

# 初始化 Vision API 客户端
vision_client = vision.ImageAnnotatorClient(credentials=credentials)

@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=3,
    max_time=30
)


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=3,
    max_time=30
)
def estimate_weight(food_name):
    """使用OpenAI估算食物重量"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """你是一个食物重量估算专家。请遵循以下规则：
                    1. 返回单人食用份量的合理重量
                    2. 普通主食（米饭、面条等）一般在200-400克之间
                    3. 肉类菜品一般在100-300克之间
                    4. 青菜类一般在100-200克之间
                    5. 水果根据大小一般在100-300克之间
                    6. 只返回数字，不要包含任何单位和文字
                    7. 确保返回的重量在合理范围内"""
                },
                {
                    "role": "user",
                    "content": f"估算一份{food_name}的重量（克），请只返回数字"
                }
            ]
        )
        
        weight_text = response.choices[0].message.content.strip()
        weight = int(''.join(filter(str.isdigit, weight_text)) or '0')
        
        # 添加合理性检查
        if weight > 1000:
            logger.warning(f"重量估算异常: {weight}克，将使用默认值")
            if any(keyword in food_name for keyword in ['饭', '面', '粥']):
                weight = 300
            elif any(keyword in food_name for keyword in ['肉', '鱼', '鸡', '鸭']):
                weight = 200
            elif any(keyword in food_name for keyword in ['菜', '青菜', '生菜']):
                weight = 150
            else:
                weight = 200
        
        return weight
        
    except Exception as e:
        logger.error(f"重量估算错误: {str(e)}")
        return 200

def detect_food(image_content):
    """使用 Google Cloud Vision API 识别食物"""
    try:
        image = vision.Image(content=image_content)
        
        # 执行标签检测
        response = vision_client.label_detection(image=image)
        labels = response.label_annotations
        
        # 执行对象检测
        objects = vision_client.object_localization(image=image).localized_object_annotations
        
        # 合并结果并筛选食物相关标签
        food_keywords = set()
        
        # 从标签中提取食物相关词
        for label in labels:
            if any(keyword in label.description.lower() for keyword in 
                  ['food', 'dish', 'cuisine', 'meal', 'fruit', 'vegetable', 'meat']):
                food_keywords.add(label.description)
        
        # 从对象中提取食物相关词
        for obj in objects:
            if any(keyword in obj.name.lower() for keyword in 
                  ['food', 'dish', 'cuisine', 'meal', 'fruit', 'vegetable', 'meat']):
                food_keywords.add(obj.name)
        
        # 如果找到食物相关标签，使用 OpenAI 翻译成中文
        if food_keywords:
            food_list = ', '.join(list(food_keywords)[:3])  # 取前三个最相关的标签
            
            # 使用 OpenAI 进行翻译和总结
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # 使用较便宜的模型
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个食物翻译专家。请将英文食物名称翻译成一个简洁的中文名称。"
                    },
                    {
                        "role": "user",
                        "content": f"请将这些食物标签翻译成一个简单的中文食物名称：{food_list}"
                    }
                ],
                max_tokens=50
            )
            
            food_name = response.choices[0].message.content.strip()
            return food_name
        else:
            raise ValueError("未能识别出食物")
            
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
        # 读取图片内容
        image_content = file.read()
        
        # 识别食物
        food_name = detect_food(image_content)
        logger.info(f"识别成功: 食物名称={food_name}")
        
        # 估算重量
        weight = estimate_weight(food_name)
        
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
        weight = request.args.get('weight')
        
        logger.info(f"收到计算卡路里请求: 食物={food_name}, 重量={weight}克")
        
        if not food_name or not weight:
            return jsonify({'error': '参数不完整', 'calories': 0}), 400
        
        # 使用OpenAI计算卡路里
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """你是一个营养专家，请根据食物重量计算卡路里。
                    1. 只返回数字，不要包含任何单位或说明
                    2. 确保返回的卡路里在合理范围内
                    3. 考虑食物的特性和烹饪方式
                    4. 如果不确定，返回相近食物的平均值"""
                },
                {
                    "role": "user",
                    "content": f"请计算{weight}克{food_name}的卡路里含量，只需要返回数字"
                }
            ]
        )
        
        calories_text = response.choices[0].message.content.strip()
        calories = int(''.join(filter(str.isdigit, calories_text)) or '0')
        
        # 添加合理性检查
        if calories > 2000:
            logger.warning(f"卡路里估算异常: {calories}，将使用默认值")
            calories = min(calories, 800)  # 单份食物不太可能超过800卡路里
            
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