from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
import os
import logging
import json  # 添加这行
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
# 添加百度其他识别接口的URL
BAIDU_DISH_DETECT_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v2/dish"
BAIDU_INGREDIENT_DETECT_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v1/classify/ingredient"
BAIDU_GENERAL_DETECT_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"

def identify_with_baidu(image_base64, access_token):
    """使用百度多个识别接口进行识别"""
    logger.info("开始食物识别流程")
    
    params = {
        'image': image_base64,
        'access_token': access_token
    }
    
    # 1. 首先尝试菜品识别
    try:
        response = requests.post(BAIDU_DISH_DETECT_URL, data=params)
        result = response.json()
        logger.info(f"菜品识别结果: {result}")
        
        if 'result' in result and len(result['result']) > 0:
            food_info = result['result'][0]
            if food_info['name'] != '非菜':
                return {
                    'name': food_info['name'],
                    'confidence': food_info['probability'],
                    'is_food': True
                }
    except Exception as e:
        logger.error(f"菜品识别出错: {str(e)}")
    
    # 2. 如果不是菜品，尝试食材识别
    try:
        logger.info("尝试食材识别")
        response = requests.post(BAIDU_INGREDIENT_DETECT_URL, data=params)
        result = response.json()
        logger.info(f"食材识别结果: {result}")
        
        if 'result' in result and len(result['result']) > 0:
            food_info = result['result'][0]
            if food_info['name'] != '非果蔬食材':
                return {
                    'name': food_info['name'],
                    'confidence': food_info['score'],
                    'is_food': True
                }
    except Exception as e:
        logger.error(f"食材识别出错: {str(e)}")
    
    # 3. 最后尝试通用物体识别
    try:
        logger.info("尝试通用物体识别")
        response = requests.post(BAIDU_GENERAL_DETECT_URL, data=params)
        result = response.json()
        logger.info(f"通用识别结果: {result}")
        
        if 'result' in result and len(result['result']) > 0:
            # 过滤出可能是食物的结果
            food_keywords = [
                '食物', '水果', '蔬菜', '零食', '饮料', '甜点', '面包', 
                '糕点', '坚果', '干果', '食材', '主食', '小吃', '糖果',
                '瓜', '果', '菜', '肉', '鱼', '虾', '蛋', '奶'
            ]
            
            first_result = result['result'][0]
            keyword = first_result['keyword']
            root = first_result.get('root', '')
            
            # 检查是否是食物相关
            is_food = any(kw in keyword or kw in root for kw in food_keywords)
            
            return {
                'name': keyword,
                'confidence': first_result['score'],
                'is_food': is_food
            }
            
    except Exception as e:
        logger.error(f"通用识别出错: {str(e)}")
    
    raise ValueError("无法识别物体")

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

def estimate_food_info_from_image(image_base64, food_name):
    """使用智谱AI同时估算食物的重量和热量"""
    try:
        logger.info(f"开始估算食物信息: {food_name}")
        
        response = client.chat.completions.create(
            model="glm-4v",
            messages=[
                {
                    "role": "system",
                    "content": """你是一个食物营养专家。请根据图片估算食物的重量和热量。

参考标准：
1. 餐盘直径一般在20-25厘米之间
2. 常见食物参考：
   - 米饭一碗：200-250克，约350卡路里
   - 肉类一份：150-200克，约250-300卡路里
   - 青菜一份：100-150克，约50卡路里
   - 水果（如苹果）：150-200克，约80卡路里

请分析后返回JSON格式数据：
{
    "weight": 数字（克）,
    "calories": 数字（卡路里）
}
只返回JSON，不要其他解释。"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"这是一张{food_name}的图片，请估算其重量和热量，直接返回JSON格式数据。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"AI响应内容: {response_text}")
        
        try:
            # 尝试解析JSON响应
            result = json.loads(response_text)
            weight = result.get('weight', 0)
            calories = result.get('calories', 0)
            
            # 合理性检查
            if not (50 <= weight <= 1000) or not (20 <= calories <= 1000):
                raise ValueError("数值超出合理范围")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"无法解析AI响应或数值异常: {str(e)}")
            # 使用默认值
            if any(keyword in food_name for keyword in ['饭', '面', '粥']):
                weight, calories = 250, 350
            elif any(keyword in food_name for keyword in ['肉', '鱼', '鸡', '鸭']):
                weight, calories = 180, 280
            elif any(keyword in food_name for keyword in ['菜', '青菜', '生菜']):
                weight, calories = 120, 50
            elif any(keyword in food_name for keyword in ['苹果', '梨', '橙子', '柚子']):
                weight, calories = 180, 80
            elif any(keyword in food_name for keyword in ['草莓', '葡萄', '樱桃']):
                weight, calories = 100, 45
            else:
                weight, calories = 200, 200
            
            logger.info(f"使用默认值 - 重量: {weight}克, 热量: {calories}卡路里")
        
        return {
            'weight': weight,
            'calories': calories
        }
        
    except Exception as e:
        logger.error(f"食物信息估算错误: {str(e)}")
        logger.error(f"错误详情: ", exc_info=True)
        return {
            'weight': 200,
            'calories': 200
        }

food_info_cache = {}

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
        image_content = file.read()
        image_base64 = base64.b64encode(image_content).decode('UTF-8')
        
        # 获取访问令牌
        access_token = get_baidu_access_token()
        
        # 使用多重识别方法
        result = identify_with_baidu(image_base64, access_token)
        
        food_name = result['name']
        confidence = result['confidence']
        is_food = result['is_food']
        
        logger.info(f"识别成功: 名称={food_name}, 置信度={confidence}, 是否食物={is_food}")
        
        response_data = {
            'name': food_name,
            'confidence': confidence,
        }
        
        if is_food:
            # 如果是食物，同时估算重量和热量
            food_info = estimate_food_info_from_image(image_base64, food_name)
            response_data.update({
                'weight': food_info['weight'],
                'calories': food_info['calories']
            })
            # 将食物信息存入缓存
            food_info_cache[food_name] = food_info
        else:
            # 如果不是食物，添加提示信息
            response_data['message'] = "这个不能吃哦"
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"识别物体时发生错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/calories', methods=['GET'])
def get_calories():
    """返回已计算的卡路里值"""
    try:
        food_name = request.args.get('foodName')
        weight = float(request.args.get('weight', 0))
        
        logger.info(f"收到计算卡路里请求: 食物={food_name}, 重量={weight}克")
        
        if not food_name or weight <= 0:
            return jsonify({'error': '参数不完整', 'calories': 0}), 400
            
        # 从缓存中获取食物信息
        food_info = food_info_cache.get(food_name)
        if food_info:
            # 根据新的重量调整卡路里值
            original_weight = food_info['weight']
            original_calories = food_info['calories']
            adjusted_calories = int((weight / original_weight) * original_calories)
            
            logger.info(f"计算结果: {adjusted_calories}卡路里 (基于原始数据: {original_calories}卡路里/{original_weight}克)")
            return jsonify({'calories': adjusted_calories})
        else:
            logger.warning(f"缓存中未找到食物信息: {food_name}")
            return jsonify({'error': '未找到食物信息', 'calories': 0}), 400
        
    except Exception as e:
        logger.error(f"计算卡路里时发生错误: {str(e)}")
        return jsonify({
            'error': str(e),
            'calories': 0
        }), 500

if __name__ == '__main__':
    port = os.getenv('PORT', 5000)
    app.run(host='0.0.0.0', port=port)