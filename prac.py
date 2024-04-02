import pandas as pd
import numpy as np
from itertools import *
import time
import multiprocessing as mp
import streamlit as st
from keras import models
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
import pickle

# 모델 학습에 사용된 데이터 열 불러오기
G1_columns = pd.read_csv('G1_train_columns.csv', index_col=0)
G2_columns = pd.read_csv('G2_train_columns.csv', index_col=0)

# model_input_columns = pd.read_csv('G1_train.csv', nrows=0).columns[:-4] # 모델에 들어가는 데이터 컬럼 (종속변수 4개 제외)
model_input_columns = G1_columns.columns[:-4]

# 원료 리스트 제작
raw_mat_list = model_input_columns.copy().tolist()
    # 규칙에 따라 정렬
custom_order = 'REQCDYAKTUPX'
custom_order_map = {char: index for index, char in enumerate(custom_order)}
def custom_sort_key(code):
    # Check if code starts with AA or XA
    if code.startswith('AA') or code.startswith('XA'):
        third_char = code[2]
        # Use the custom order map to get the sort weight of the third character
        # Use a default value that sorts these codes after the specified characters if the third character is not in custom_order
        sort_weight = custom_order_map.get(third_char, len(custom_order_map))
    else:
        # For other codes, use a default sort logic
        # Here, we sort them alphabetically and place them after the AA/XA codes
        sort_weight = len(custom_order_map) + 1
    # Return a tuple that includes the sort weight and then the full code to ensure unique sorting
    return (sort_weight, code)
raw_mat_list = sorted(raw_mat_list, key=custom_sort_key)

st.title('SVCD')
st.header('Recipe Range Setting (Required)')
# st.subheader('Variables')
st.write('Please set phr range for each raw material.')
st.caption('If polymer contains oil, please use the phr value excluding the oil content.')

raw_mat_slider = st.multiselect('Raw Material List', raw_mat_list, key='2')

col = raw_mat_slider.copy()
base_rm = list()
base_phr = list()
phr_min = list()
phr_max = list()
rubber_col = list()

stp = st.number_input('Step', step=1)
st.caption('"Step" must be greater than 1. Higher "Step" value increases the number of data points within the range.')

# 슬라이더 생성 (키보드 입력할 수 있는 기능 추가)
for i in raw_mat_slider:
    # 범위 초기화
    # range_val = st.slider(f"{i} Range", min_value=0.0, max_value=120.0, value=(0.0, 120.0), step=0.1, key=f"{i}_range")
    
    with st.expander(i, expanded=True): # i: AAE325A, AAE333A, ...
        minv, maxv = st.columns(2)
        with minv:
            min_val = st.number_input(f"Minimum {i}", min_value=0.0, max_value=150.0, value=0.0, step=0.01, key=f"{i}_min")
        with maxv:
            max_val = st.number_input(f"Maximum {i}", min_value=0.0, max_value=150.0, value=150.0, step=0.01, key=f"{i}_max")
    st.markdown("###") # 사이사이 공간 만들어주는 용도

    # if (min_val, max_val) != range_val:
    #     range_val = (min_val, max_val)

    # min, max 같을 때 처리
    # if min_val == max_val:
    #     base_rm.append(i)
    #     base_phr.append(min_val)
    # else:
    phr_min.append(min_val)
    phr_max.append(max_val)


st.header('Fixed Material Setting (Optional)')
# st.subheader('Fixed phr Settings')
st.write('Please specify raw materials with fixed phr values.')
fixed_phr_dict = {}  # Dictionary to store fixed phr values
fixed_phr_materials = st.multiselect('Raw Material List', raw_mat_list, key='fixed_phr')
for material in fixed_phr_materials:
    fixed_phr_value = st.number_input(f"{material}", min_value=0.0, max_value=200.0, key=f"fixed_{material}")
    fixed_phr_dict[material] = fixed_phr_value

col = fixed_phr_materials + col

st.header('Reference Recipe Setting (Optional)')
# st.subheader('Variables')
st.write('Please set phr values for baseline recipe.')

reference_slider = st.multiselect('Raw Material List', raw_mat_list, key='1')
reference_recipe = {}
for i in reference_slider:
    value = st.number_input(i)
    reference_recipe[i] = value



# # 슬라이더 생성
# for i in range(len(raw_mat_slider)):

#     phr = st.slider(raw_mat_slider[i], 0.0,120.0,(0.0,120.0), step=0.1)
#     if phr[0] == phr[1]:
#         base_rm.append(col.pop(col.index(raw_mat_slider[i])))
#         base_phr.append(phr[0])
#     else: 
#         phr_min.append(phr[0])
#         phr_max.append(phr[1])

# 고정값 phr 저장
for i in fixed_phr_materials:
    phr_min.insert(0,fixed_phr_dict[i])
    phr_max.insert(0,fixed_phr_dict[i])
    # if i[2] == 'E' or i[2] == 'Q' or i[2]=='R':
    #     rubber_col.append(i)





# 불러오기
with open('oil_content_dict.pickle', 'rb') as f:
    oil_content_dict = pickle.load(f)

# not_in_oil_content = (set(pd.read_csv('G1_train.csv', nrows=0).columns[:-4]) - set(oil_content_dict.keys()))

not_in_oil_content = (set(G1_columns.columns[:-4]) - set(oil_content_dict.keys()))

for key in not_in_oil_content:
    oil_content_dict[key] = 0

oil_content = []

if st.button('Create Recipes'):
    real_start = int(time.time())
    # rubber col에 3번째 알파벳이 Q나 E나 R인 애들 material code 저장
    for i in col:
        if i[2] == 'E' or i[2] == 'Q' or i[2]=='R':
            rubber_col.append(i)
            oil_content.append(oil_content_dict[i])

    # 고무 재료 중 마지막거 잠깐 빼놓고 인덱스 저장
    rubber_tmp = rubber_col[:-1]
    rubber_idx = col.index(rubber_col[-1])

    col.pop(rubber_idx)

    print('rubber index:', rubber_idx)

    phr_min_tmp = phr_min.pop(rubber_idx) # phr_min_tmp: rubber 컬럼에 대한 phr lower limit
    phr_max_tmp = phr_max.pop(rubber_idx) # phr_max_tmp: rubber 컬럼에 대한 phr upper limit

    # 사용자가 설정한 범위를 stp만큼 분할
    df_range = pd.DataFrame(columns=col)

    for i, phr in enumerate(list(zip(phr_min, phr_max))):
        phr_range = np.linspace(list(phr)[0], list(phr)[1],stp)
        df_range.iloc[:,i] = phr_range

# 시간 측정 (구간 1)
    # product 함수로 모든 경우의 수 리스트 생성
    initial_time = time.time()
    df_index = pd.DataFrame(columns=col)#, index=list(range(stp**len(col))))
    idx = list(product(range(stp),repeat=len(col)))
    print(f"elapsed time for process 1: {round((time.time() - initial_time),2)} sec ({round((time.time()-initial_time)/60, 2)} min)")
    start = time.time()


# 시간 측정 (구간 2)
    # start = time.time()
    # 모든 경우의 수 조합 담는 데이터프레임 생성
    # for i in idx:
        # df_index.loc[len(df_index),:] = list(i)

    # 수행 시간 감소
    # # 1.
    # # 모든 경우의 수를 담을 사이즈의 빈 데이터프레임 생성해놓고
    # df_index = pd.DataFrame(index=range(len(idx)), columns=df_range.columns)

    # # 채우기
    # for i, combination in enumerate(idx):
    #     df_index.iloc[i] = combination

    # 2.
    df_index = pd.DataFrame(idx, columns = df_range.columns)
    

    print(f"elapsed time for process 2: {(round((time.time() - start),2))} sec ({round((time.time()-start)/60, 2)} min)")

# 시간 측정 (구간 3)
    # range에 있는 값들을 경우의수에 맞게 recipe로 옮기는 과정
    # start = time.time()
    df_recipe = pd.DataFrame(columns=col, index=df_index.index)
    # map 함수로 range에 있는 값들을 경우의수에 맞게 recipe로 옮김 
    for col in df_range.columns:
        df_recipe[col] = df_index[col].map(df_range[col])

    # for i in df_index.index:
    #     for k in range(len(df_range.columns)):
    #         df_recipe.iloc[i,k] = df_range.iloc[df_index.iloc[i,k],k]
            # print(f'df_index.index: {df_index.shape[0]}, df_range.columns: {df_range.shape[1]}')
            # print(i,k)
    print(f"elapsed time for process 3: {(round((time.time() - start),2))} sec ({round((time.time()-start)/60, 2)} min)")
    print(f"total time taken: {round(time.time()-initial_time, 2)} sec ({round((time.time()-initial_time)/60, 2)} min)")

    df_recipe.drop_duplicates(inplace=True) # 중복 제거

    print('rubber_col:', rubber_col)

        # 고무 재료가 하나만 있으면 100phr에 상응하는 값으로 다 채워줌
    if len(rubber_col) == 1:
        single_rubber_proportion = (100 + oil_content[0])
        df_recipe[rubber_col[0]] = single_rubber_proportion

        # 잠깐 빼뒀던 마지막 폴리머 컬럼을 원래 있던 자리에 넣는 행위
        # 값: 100 - (다른 rubber phr 합)
        # 여기서 rubber column에 있는 다른애들 phr 합이 100을 넘어가는 경우가 있음 > 값이 음수가 됨
    else:
        df_recipe.insert(len(rubber_col)-1, rubber_col[-1], (100 - (df_recipe[rubber_col[:-1]]/[100+i for i in oil_content[:-1]]*100).sum(axis=1)) * (100 + oil_content[-1])/100)

    # 고정값 받은대로 설정
    for material, value in fixed_phr_dict.items():
        if material in df_recipe.columns:
            df_recipe[material] = value

    # # 범위 안에 들어오는지 확인
    #     # phr max보다 rubber phr이 큰 경우 여기서 잘려나감
    #     # 고무 재료가 하나만 있고 100+oil_content가 상한을 넘어버리면, df_recipe가 비어버리는 상황이 발생
    #     # 그래서 하나만 있을때는 일단 확인안하도록 해놓음 (검토해서 수정해야할듯)

    if len(rubber_col) > 1:
        df_recipe = df_recipe[df_recipe[rubber_col[-1]] >= phr_min_tmp]
        df_recipe = df_recipe[df_recipe[rubber_col[-1]] <= phr_max_tmp]
    
    df_recipe[base_rm] = base_phr
    
    # AAD113A랑 AAD342A가 둘다 포함되어 있으면 AAD113A는 AAD342A의 8%로 설정해줌
    # if ('AAD113A' in df_recipe.columns) & ('AAD342A' in df_recipe.columns):
    if 'AAD342A' in df_recipe.columns:
        df_recipe['AAD113A'] = df_recipe['AAD342A']*0.08
    
    df_recipe.drop_duplicates(inplace=True)
    df_recipe.reset_index(inplace=True, drop=True)
    print(df_recipe.shape)
    df_recipe = df_recipe.astype(float)

    df_recipe.to_csv("recipe.csv", index=False) # 레시피 저장



    # print("***run time(sec) :", int(time.time()) - start)

    # 모델 학습 데이터랑 같은 형태로 만들어주기 위해서, 학습데이터에 있는데 recipe에 없는 재료는 0으로 채워주는 과정
    # df_train = pd.read_csv('G1_P42X_train.csv')
    # model_input_columns = df_train.columns[:-4] # 모델에 들어가는 데이터 컬럼 (종속변수 4개 제외)
    # model_input_columns = pd.read_csv('G1_P42X_train.csv', nrows=0).columns[:-4]
    # model_output_columns_g1 = pd.read_csv('G1_train.csv', nrows=0).columns[-4:] # 아웃풋 되는 물성 컬럼
    # model_output_columns_g2 = pd.read_csv('G2_train.csv', nrows=0).columns[-4:]

    model_output_columns_g1 = G1_columns.columns[-4:] # 아웃풋 되는 물성 컬럼
    model_output_columns_g2 = G2_columns.columns[-4:]

    zeros = np.zeros([df_recipe.shape[0], len(model_input_columns)])
    df_recipe_filled = pd.DataFrame(zeros, columns = model_input_columns)
    df_recipe_filled[df_recipe.columns] = df_recipe
    # Reference recipe 추가
    df_reference = pd.DataFrame([np.zeros(len(model_input_columns))], columns= model_input_columns)
    df_reference[list(reference_recipe.keys())] = list(reference_recipe.values())
    df_recipe_filled = pd.concat([df_reference, df_recipe_filled], ignore_index=True)

    # df_train_g2 = pd.read_csv('G2_train.csv')
    # model_input_columns_g2 = df_train_g2.columns[:-4] # 모델에 들어가는 데이터 컬럼 (종속변수 4개 제외)
    # model_output_columns_g2 = df_train_g2.columns[-4:] # 아웃풋 되는 물성 컬럼
    # zeros = np.zeros([df_recipe.shape[0], len(model_input_columns_g2)])
    # df_recipe_filled_g2 = pd.DataFrame(zeros, columns = model_input_columns_g2)
    # df_recipe_filled_g2[df_recipe.columns] = df_recipe

    # SVCD 모델 로딩
    # G1
    json_file = open('viscoelastic_240307_total_best_structure_g1.json', 'r')
    loaded_model_json_g1 = json_file.read()
    json_file.close()

    # G2
    json_file = open('viscoelastic_240307_total_best_structure_g2.json', 'r')
    loaded_model_json_g2 = json_file.read()
    json_file.close()




    # ReLU와 유사한 activation function (Swish, or SiLU)
    def _swish(x):
        return K.sigmoid(x) * x

    loaded_model_g1 = model_from_json(loaded_model_json_g1, custom_objects={'swish': _swish})
    loaded_model_g2 = model_from_json(loaded_model_json_g2, custom_objects={'swish': _swish})

    loaded_model_g1.load_weights('viscoelastic_240307_total_best_weights_g1.h5')
    loaded_model_g2.load_weights('viscoelastic_240307_total_best_weights_g2.h5')
    # print(loaded_model_g1.summary())
    # print(loaded_model_g2.summary())


    # 예측 결과 저장
    pred_g1 = loaded_model_g1.predict(df_recipe_filled)
    pred_g1_df = pd.DataFrame(pred_g1, columns=model_output_columns_g1)
    # pred_g1_df.to_csv("pred_g1.csv", index=False)

    st.success(f'Step 1/3 completed, time taken: {round(time.time() - real_start, 2)} sec.')
    start = time.time()


    pred_g2 = loaded_model_g2.predict(df_recipe_filled)
    pred_g2_df = pd.DataFrame(pred_g2, columns=model_output_columns_g2)
    # pred_g2_df.to_csv("pred_g2.csv", index=False)

    pred_tand = pred_g2 / pred_g1
    pred_tand_df = pd.DataFrame(pred_tand, columns=['tand_-30', 'tand_0', 'tand_25', 'tand_60'])

    st.success(f'Step 2/3 completed, time taken: {round(time.time() - start, 2)} sec.')
    start = time.time()

### 역설계
    ### 레퍼런스 레시피 받아서 예측값 사용
    # 레퍼런스 안들어오면 제한 없음
    if df_reference.sum().sum() == 0:
        G2_0_limit = -999999
        tand_60_limit = 9999999999999
    # 들어오면 레퍼런스 예측물성보다 좋도록 제한
    else:
        G2_0_limit = pred_g2_df.loc[0,'G2_0'] # 0'C g2 최소값 설정
        tand_60_limit = pred_tand_df.loc[0,'tand_60'] # 60'C tand 최대값 설정
    print(df_reference)
    print('G2 lim:', G2_0_limit)
    print('tand lim:', tand_60_limit)

    df_G1 = pred_g1_df.copy()
    df_G2 = pred_g2_df.copy()
    test_input_file_name = 'recipe.csv'

    df_G1.columns = [-30,0,25,60]
    df_G2.columns = [-30,0,25,60]
    df_G2_0 = df_G2[0].sort_values(ascending=False) # 0'C g2가 높은 순으로 정렬

    df_tand = df_G2.div(df_G1) # tand 데이터 생성
    df_tand_60 = df_tand[60].sort_values() # 60'C tand 낮은 순으로 정렬

    df_G2_0 = df_G2_0[df_G2_0>=G2_0_limit]
    df_tand_60 = df_tand_60[df_tand_60<=tand_60_limit] # 설정 limit 벗어나면 제거


    df_G1.columns = ['G1_-30','G1_0','G1_25','G1_60']
    df_G2.columns = ['G2_-30','G2_0','G2_25','G2_60']
    df_tand.columns = ['tand_-30','tand_0','tand_25','tand_60']

    #
    df_recipe_filled_0rmv = df_recipe_filled.copy()
    df_recipe_filled_0rmv = df_recipe_filled_0rmv.loc[:, ~(df_recipe_filled_0rmv == 0).all()]

    # Ver 1. 기존 순위 산출 알고리즘
    # start = time.time()
    if len(df_G2_0.index) > len(df_tand_60.index):
        bigger_index = df_G2_0.index
        smaller_index = df_tand_60.index
    else:
        bigger_index = df_tand_60.index
        smaller_index = df_G2_0.index

    idx = list()
    num = 0
    for i in range(len(bigger_index)):
        if bigger_index[i] in smaller_index[:i+1]:
            # print(i)
            # print(bigger_index[i])
            # print(list(smaller_index).index(bigger_index[i]))
            # print(smaller_index[list(smaller_index).index(bigger_index[i])])
            num += 1
            if num == 10:
                std = i
                # print(f'std:{std}')
                for j in range(std+1):
                    if bigger_index[j] in smaller_index[:std+1]:
                        # print(f'j:{j}')
                        # print(f'bigger_index[j]: {bigger_index[j]}')
                        # print(f'list(smaller_index).index(bigger_index[j]): {list(smaller_index).index(bigger_index[j])}')
                        # print(f'smaller_index[list(smaller_index).index(bigger_index[i])]: {smaller_index[list(smaller_index).index(bigger_index[i])]}')
                        idx.append(bigger_index[j])
                break
    # print(std)
    # print(f'idx: {idx}')
    # print(len(idx))
    # print(len(set(idx)))
    print('elapsed time, Ver 1:', round((time.time() - start), 4))

    print(df_G1.columns)

    # Ver 2. 정규화 후 순위별로 점수 산출, 합산 -- 이상치에 민감
    # start = time.time()
    scaler = MinMaxScaler()
    df_rank_scaling = pd.DataFrame([df_tand_60, df_G2_0]).T
    G2_normalized = scaler.fit_transform(np.array(df_rank_scaling[0]).reshape(-1, 1))
    tand_normalized = 1 - scaler.fit_transform(np.array(df_rank_scaling[60]).reshape(-1, 1))
    avg_score = (G2_normalized + tand_normalized) / 2
    df_rank_scaling['Score'] = avg_score
    df_rank_scaling = pd.concat([df_rank_scaling['Score'], df_recipe_filled_0rmv, df_G1, df_G2, df_tand], axis=1)
    df_rank_scaling = df_rank_scaling.sort_values('Score', ascending=False)
    df_rank_scaling = df_rank_scaling.dropna()
    # 순위 추가
    rank_scaling = df_rank_scaling['Score'].rank(method='min', ascending=False)
    df_rank_scaling.insert(0, 'Rank', rank_scaling)

    print('elapsed time, Ver 2:', round((time.time() - start), 4))

    # Ver 3. 각 물성별 순위 자체를 기준으로 점수 산출
    # 교집합 탐색
    common_indices = df_G2_0.index.intersection(df_tand_60.index)

    # G2와 tand 둘다 값을 가지는 데이터만 골라냄
    df_G2_common = df_G2_0.loc[common_indices]
    df_tand_common = df_tand_60.loc[common_indices]

    # 순위 산출
    df_G2_ranks = df_G2_common.rank(ascending=False) # 높을수록 좋음
    df_tand_ranks = df_tand_common.rank(ascending=True) # 낮을수록 좋음

    # 등수 합산
    composite_scores = df_G2_ranks + df_tand_ranks

    # (tand값, G2값, 점수) 포함하는 데이터프레임 생성
    sorted_scores = composite_scores.sort_values()
    df_composite_rank = pd.concat([df_recipe_filled_0rmv, df_G1.loc[sorted_scores.index], df_G2.loc[sorted_scores.index], 
                                   df_tand.loc[sorted_scores.index]], axis=1)
    df_composite_rank.insert(0, 'Score', sorted_scores)
    df_composite_rank = df_composite_rank.dropna()
    
    # 순위 추가
    rank_composite = df_composite_rank['Score'].rank(method='min', ascending=True)
    df_composite_rank.insert(0, 'Rank', rank_composite)
    df_composite_rank = df_composite_rank.sort_values('Score', ascending=True)

    # pd.DataFrame({
    #     'tand_60': df_tand_60.loc[sorted_scores.index],
    #     'G2_0': df_G2_0.loc[sorted_scores.index],
    #     'score': sorted_scores
    # })
    # df_composite_rank = pd.concat([df_composite_rank, df_recipe_filled_0rmv], axis=1)
    ###

    df_G1.columns = ['G1_-30','G1_0','G1_25','G1_60']
    df_G2.columns = ['G2_-30','G2_0','G2_25','G2_60']
    df_tand.columns = ['tand_-30','tand_0','tand_25','tand_60']
    df = pd.read_csv(f'{test_input_file_name}') # recipe.csv
    # df['Plasticizer']=df[['AAP501A','AAT231A']].sum(axis=1)+df['AAE325A']*0.2
    # df['Pla_Sil_Ratio'] = df['AAD342A']/df['Plasticizer']
    # df['Tg_calc'] = (df['AAE325A']*0.8*(-50)+df['AAQ233A']*(-92)+48*df['AAT231A']-101*df['AAP501A'])/df[['AAE325A','AAQ233A','AAT231A','AAP501A']].sum(axis=1)
    # df['Wet Index_New'] = (df['AAD342A']**1.8)/((df['Tg_calc'].abs())**0.8)

    input = pd.concat([df_recipe_filled_0rmv, df_G1, df_G2, df_tand], axis=1)

    input.iloc[df_tand.index].to_csv('target_recipe_total_2_test.csv')

    target_recipe = input.iloc[idx]
    target_recipe.insert(0, 'Rank', range(1,target_recipe.shape[0]+1))
    # target_recipe.loc[:,'rank'] = range(1,target_recipe.shape[0]+1)
    # target_recipe = target_recipe.iloc[:,:-12]
    target_recipe.to_csv('target_recipe_2_test.csv')
    st.success(f'Step 3/3 completed, time taken: {round(time.time() - start, 2)} sec.')
    st.success(f'All steps completed, total time taken: {round(time.time() - real_start, 2)} sec.')

### 결과 출력
    x = st.expander('Recipe Information', expanded=True)

    x.caption('Index 0 refers to the reference recipe.')
    x.caption('The phr values for polymers displayed incorporate both the polymer and its inherent oil content.')

    x.subheader('Recipe Ranking')
    x.write('Method 1. Original Algorithm')
    x.dataframe(target_recipe, width = 800)

    x.write('Method 2. Min-max Scaling')
    x.dataframe(df_rank_scaling, width = 800)

    x.write('Method 3. Composite Ranking')
    x.dataframe(df_composite_rank, width = 800)
    
    x.subheader('All recipes')
    x.write('{} recipes were created'.format(df_recipe_filled_0rmv.shape[0]))
    df_recipe_filled_concat = pd.concat([df_recipe_filled_0rmv, pred_g1_df, pred_g2_df, pred_tand_df], axis=1)
    
    # 기존 버전
    x.dataframe(df_recipe_filled_concat, width = 800)
    
    # # 레퍼런스 색칠 버전 (오래 걸림)
    # def highlight_first_row(s):
    #     return ['background-color: yellow' if s.name == 0 else '' for v in s]

    # pd.set_option("styler.render.max_elements", 1308300)
    
    # styled_df = df_recipe_filled_concat.style.apply(highlight_first_row, axis=1)    
    # st.write(styled_df, unsafe_allow_html=True)
    
    # x.write('###### 예측 물성')
    # x.write("G'")
    # x.dataframe(pred_g1_df, width = 800)

    # x.write("G''")
    # x.dataframe(pred_g2_df, width = 800)

    # x.write("Tan Delta")
    # x.dataframe(pred_tand_df, width = 800)


    # print(target_recipe.shape[0])
    # print(df_rank_scaling.shape[0])
    # print(df_composite_rank.shape[0])
    # print(pred_g1_df.loc[0,:])
    # print(df_recipe.columns)
