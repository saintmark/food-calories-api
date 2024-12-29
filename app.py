from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
import os
import logging
from zhipuai import ZhipuAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 百度AI配置
BAIDU_API_KEY = os.getenv('BAIDU_API_KEY')
BAIDU_SECRET_KEY = os.getenv('BAIDU_SECRET_KEY')
BAIDU_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
BAIDU_DISH_DETECT_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v2/dish"

# 智谱AI配置
ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY')
if not ZHIPU_API_KEY:
    raise ValueError("未找到ZHIPU_API_KEY环境变量")

logger.info(f"ZHIPU_API_KEY: {ZHIPU_API_KEY[:10]}...")
client = ZhipuAI(api_key=ZHIPU_API_KEY)

def get_baidu_access_token():
    """获取百度AI访问令牌"""
    params = {
        'grant_type': 'client_credentials',
        'client_id': BAIDU_API_KEY,
        'client_secret': BAIDU_SECRET_KEY
    }
    response = requests.post(BAIDU_TOKEN_URL, params=params)
    return response.json().get('access_token')

def estimate_weight(food_name):
    """使用智谱AI估算食物重量"""
    try:
        weight_response = client.chat.completions.create(
            model="glm-4",
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
        
        weight_text = weight_response.choices[0].message.content.strip()
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
        # 读取图片并转换为base64
        image_base64 = base64.b64encode(file.read()).decode('UTF-8')
        
        # 调用百度AI识别菜品
        access_token = get_baidu_access_token()
        params = {
            'image': image_base64,
            'access_token': access_token
        }
        
        response = requests.post(BAIDU_DISH_DETECT_URL, data=params)
        logger.info(f"百度API返回结果: {response.json()}")
        
        result = response.json()
        
        if 'result' in result and len(result['result']) > 0:
            food_name = result['result'][0]['name']
            confidence = result['result'][0]['probability']
            
            logger.info(f"识别成功: 食物名称={food_name}, 置信度={confidence}")
            
            # 使用智谱AI估算食物重量
            weight = estimate_weight(food_name)
            
            return jsonify({
                'name': food_name,
                'confidence': confidence,
                'weight': weight
            })
        
        logger.error("无法识别食物")
        return jsonify({'error': '无法识别食物'}), 400
        
    except Exception as e:
        logger.error(f"识别食物时发生错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/calories', methods=['GET'])
def get_calories():
    """使用智谱AI估算卡路里"""
    try:
        food_name = request.args.get('foodName')
        weight = request.args.get('weight')
        
        logger.info(f"收到计算卡路里请求: 食物={food_name}, 重量={weight}克")
        
        if not food_name or not weight:
            return jsonify({'error': '参数不完整', 'calories': 0}), 400
        
        response = client.chat.completions.create(
            model="glm-4",
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