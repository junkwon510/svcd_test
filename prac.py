import pandas as pd
import numpy as np
import os
import re
from itertools import *
import time
import multiprocessing as mp
import streamlit as st
from keras import models
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
import pickle



def reorder_columns(df):
    matcol = [col for col in df.columns if col not in (['Rank'] + list(df_G1.columns) + list(df_G2.columns) + list(df_tand.columns))]
    columns_sorted = sorted(matcol, key=custom_sort_key)
    if 'Rank' in df.columns:
        new_order = ['Rank'] + columns_sorted + list(df_G1.columns) + list(df_G2.columns) + list(df_tand.columns)
    else:
        new_order = columns_sorted + list(df_G1.columns) + list(df_G2.columns) + list(df_tand.columns)
    df = df[new_order]
    return df

tab_1, tab_2, tab_3, tab_4 = st.tabs(['SVCD', 'Guideline', 'Updates', 'Suggestions'])


with tab_1:
    # 모델 학습에 사용된 데이터 열 불러오기
    G1_columns = pd.read_csv('G1_train_columns.csv', index_col=0)
    G2_columns = pd.read_csv('G2_train_columns.csv', index_col=0)

    # model_input_columns = pd.read_csv('G1_train.csv', nrows=0).columns[:-4] # 모델에 들어가는 데이터 컬럼 (종속변수 4개 제외)
    model_input_columns = G1_columns.columns[:-4]

    # Oil content dictionary 불러오기
    with open('oil_content_dict.pickle', 'rb') as f:
        oil_content_dict = pickle.load(f)

    # not_in_oil_content = (set(pd.read_csv('G1_train.csv', nrows=0).columns[:-4]) - set(oil_content_dict.keys()))

    not_in_oil_content = (set(G1_columns.columns[:-4]) - set(oil_content_dict.keys()))

    for key in not_in_oil_content:
        oil_content_dict[key] = 0

    # RM PJT SSBR 정보 참고하여 딕셔너리 수정
    for key, value in oil_content_dict.items():
        if value == 4.9:
            oil_content_dict[key] = 5.0
        elif value == 36.9:
            oil_content_dict[key] = 37.5
        elif (value == 49.2 or value == 49.9):
            oil_content_dict[key] = 50.0

    oil_content_dict['AAE525A'] = 25.0

    oil_content = []

    # 학습에 사용된 모든 원료 리스트
    raw_mat_list = model_input_columns.copy().tolist()

    # 규칙에 따라 정렬하는 함수
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
    st.header('Recipe Range Setting (required)')
    # st.subheader('Variables')
    st.write('Please set phr range for each raw material.')
    st.write('If polymer contains oil, please use phr value excluding the oil content.')

    raw_mat_slider = st.multiselect('Raw Material List', raw_mat_list, key='2')

    temp_list = []
    for i in raw_mat_slider:
        if (i[2]!='R') and (i[2]!='E') and (i[2]!='Q'):
            temp_list.append(1)
        else:
            temp_list.append(0)
    if 0 not in temp_list:
        st.warning('No rubber in raw material list. Please add rubber above, or below in the fixed material setting.')

    raw_mat_col = raw_mat_slider.copy() # 기존 range setting에서 받은것만 담아놓을 애
    col = raw_mat_slider.copy() # 나중에 fixed까지 추가해줄 애

    base_rm = list()
    base_phr = list()
    phr_min = list()
    phr_max = list()
    rubber_col = list()
    fixed_rubber_col = list()
    raw_mat_rubber_col = list()

    # 슬라이더 생성 (키보드 입력할 수 있는 기능 추가)
    for i in raw_mat_slider:
        # 범위 초기화
        # range_val = st.slider(f"{i} Range", min_value=0.0, max_value=120.0, value=(0.0, 120.0), step=0.1, key=f"{i}_range")
        
        with st.expander(i, expanded=True): # i: AAE325A, AAE333A, ...
            minv, maxv = st.columns(2)
            with minv:
                min_val = st.number_input(f"Minimum {i}", min_value=0.0, max_value=120.0, value=0.0, step=0.01, key=f"{i}_min")
            with maxv:
                max_val = st.number_input(f"Maximum {i}", min_value=0.0, max_value=120.0, value=120.0, step=0.01, key=f"{i}_max")
        st.markdown("###") # 사이사이 공간 만들어주는 용도

        # Range Setting에서 min, max 같을 때 메시지 띄워주기
        # Fixed에서 넣어주면 문제 없는데, range setting에서 같게하면 문제 발생
        # (ex. oil 있는 고무 100으로 고정하고 다른 고무 넣으면 다른 고무에 phr 생김)
        if min_val == max_val:
            st.warning('Minimum and maximum values are the same. Please use "Fixed Material Setting" below.')

        # if (min_val, max_val) != range_val:
        #     range_val = (min_val, max_val)

        # min, max 같을 때 처리
        # if min_val == max_val:
        #     base_rm.append(i)
        #     base_phr.append(min_val)
        # else:
        phr_min.append(min_val)
        phr_max.append(max_val)

    

    
    stp = st.number_input('Step', min_value = 0, step=1)

    st.caption('"Step" must be greater than 1. Higher "Step" value increases the number of data points within the range.')

    if stp>0 and stp**((len(raw_mat_col)-1))>=1000000:
        st.warning('Number of recipes created may be too large.')

    st.header('Fixed Material Setting (optional)')
    # st.subheader('Fixed phr Settings')
    st.write('Please specify raw materials with fixed phr values.')
    fixed_phr_dict = {}  # Dictionary to store fixed phr values
    fixed_col = st.multiselect('Raw Material List', raw_mat_list, key='fixed_phr')
    for material in fixed_col:
        fixed_phr_value = st.number_input(f"{material}", min_value=0.0, max_value=200.0, key=f"fixed_{material}")
        if material[2] == 'E' or material[2] == 'Q' or material[2]=='R':
            fixed_phr_value = fixed_phr_value*oil_content_dict[material]/100 + fixed_phr_value
        fixed_phr_dict[material] = fixed_phr_value

    col = fixed_col + col

    st.header('Reference Recipe Setting (optional)')
    # st.subheader('Variables')
    st.write('Please set phr values for baseline recipe.')

    ###
    #cpd_list = ['CQP07', 'CQP08', 'CQP09','CQP11', 'CQP19', 'CQP22', 'CQP25', 'CQP27',
    #            'CQP28', 'CQP37', 'CQP47','CQP50', 'CQP53', 'CQP59', 'CQP60', 'CQP70']
    #compound_slider = st.multiselect('Compound List', cpd_list, key='3')
    ###

    reference_slider = st.multiselect('Raw Material List', raw_mat_list, key='1')
    reference_recipe = {}
    for i in reference_slider:
        value = st.number_input(i)
        reference_recipe[i] = value


    phr_min_rawmat = phr_min.copy()
    phr_max_rawmat = phr_max.copy()

    # 고정값 phr 저장
    for i in fixed_col:
        phr_min.insert(0,fixed_phr_dict[i])
        phr_max.insert(0,fixed_phr_dict[i])
        # if i[2] == 'E' or i[2] == 'Q' or i[2]=='R':
        #     rubber_col.append(i)


    if st.button('Create Recipes'):
        st.success('Initiate recipe creation.')
        real_start = int(time.time())
        start = int(time.time())
        # rubber col에 3번째 알파벳이 Q나 E나 R인 애들(폴리머) material code 저장
        for i in fixed_col:
            if i[2] == 'E' or i[2] == 'Q' or i[2]=='R':
                fixed_rubber_col.append(i)
                oil_content.append(oil_content_dict[i])
        for i in raw_mat_col:
            if i[2] == 'E' or i[2] == 'Q' or i[2]=='R':
                raw_mat_rubber_col.append(i)
                oil_content.append(oil_content_dict[i])

        
        
        # 고무 재료 중 마지막거 잠깐 빼놓고 인덱스 저장
        if raw_mat_rubber_col!=[]:
            rubber_excluded = raw_mat_rubber_col[-1]
            rubber_tmp = raw_mat_rubber_col[:-1] # 마지막 폴리머 제외한 것
            rubber_idx = raw_mat_rubber_col.index(raw_mat_rubber_col[-1]) # 마지막 폴리머 인덱스

            # raw_mat_rubber_col.pop(rubber_idx) # 마지막 폴리머 제거

            phr_min_tmp = phr_min_rawmat.pop(rubber_idx) # 마지막 폴리머 phr_min 제거, 저장. phr_min_rawmat은 마지막 고무 빠지게 됨
            phr_max_tmp = phr_max_rawmat.pop(rubber_idx) # 마지막 폴리머 phr_max 제거, 저장


        # 사용자가 설정한 범위를 stp만큼 분할
            raw_mat_col_filtered = [item for item in raw_mat_col if item != rubber_excluded]
        else:
            raw_mat_col_filtered = raw_mat_col.copy()

        # 고정 폴리머 phr 합에 따라 가변 폴리머 phr 제한하려고 함
        phr_fixed_rubber = []
        if fixed_rubber_col != []:
            for i in fixed_rubber_col:
                phr_fixed_rubber.append(fixed_phr_dict[i])

            # (100 - 고정 폴리머 phr합)을 maximum으로 설정
            total_phr_fixed_rubbers = sum(phr_fixed_rubber)
            max_phr_for_variable_rubbers = 100 - total_phr_fixed_rubbers

            # phr_max_rawmat 값중에 고무 phr에 해당하는 값(raw_mat_col_filtered에서 고무 순서랑 같은)만 max_phr_for_variable_rubbers로 바꿔줌
            phr_max_rawmat = [max_phr_for_variable_rubbers if mat[2] in ['E', 'Q', 'R'] 
                            else max_val for mat, max_val in zip(raw_mat_col_filtered, phr_max_rawmat)]

        df_range = pd.DataFrame(columns=raw_mat_col_filtered)
        for i, phr in enumerate(list(zip(phr_min_rawmat, phr_max_rawmat))):
                phr_range = np.linspace(list(phr)[0], list(phr)[1],stp)
                df_range.iloc[:,i] = phr_range


    # 시간 측정 (구간 1)
        # product 함수로 모든 경우의 수 리스트 생성
        initial_time = time.time()

        idx = list(product(range(stp),repeat=len(raw_mat_col)))
        print(f"elapsed time for process 1: {round((time.time() - initial_time),2)} sec ({round((time.time()-initial_time)/60, 2)} min)")
        start = time.time()

        df_index = pd.DataFrame(idx, columns = raw_mat_col)
        

        print(f"elapsed time for process 2: {(round((time.time() - start),2))} sec ({round((time.time()-start)/60, 2)} min)")

    # 시간 측정 (구간 3)
        # range에 있는 값들을 경우의 수에 맞게 recipe로 옮기는 과정
        # start = time.time()
        # if rubber_excluded is defined, exclude it from the columns
        

        if raw_mat_rubber_col!=[]:
            df_recipe = pd.DataFrame(columns=[item for item in col if item != rubber_excluded], index=df_index.index)

        else:
            df_recipe = pd.DataFrame(columns=col, index=df_index.index)
        # map 함수로 range에 있는 값들을 경우의수에 맞게 recipe로 옮김 

        for col in raw_mat_col_filtered:
            df_recipe[col] = df_index[col].map(df_range[col])

        # for i in df_index.index:
        #     for k in range(len(df_range.columns)):
        #         df_recipe.iloc[i,k] = df_range.iloc[df_index.iloc[i,k],k]
                # print(f'df_index.index: {df_index.shape[0]}, df_range.columns: {df_range.shape[1]}')
                # print(i,k)
        print(f"elapsed time for process 3: {(round((time.time() - start),2))} sec ({round((time.time()-start)/60, 2)} min)")
        print(f"total time taken: {round(time.time()-initial_time, 2)} sec ({round((time.time()-initial_time)/60, 2)} min)")

        df_recipe.drop_duplicates(inplace=True) # 중복 제거

        rubber_col = fixed_rubber_col + raw_mat_rubber_col

        # fixed가 들어왔으면, 지금 df_recipe에서 fixed rubber이 Nan으로 되어있어서 채워줌
        if fixed_rubber_col != []:
            for i in fixed_rubber_col:
                df_recipe[i] = fixed_phr_dict[i]


        # 고무 재료가 하나만 있으면 100phr에 상응하는 값으로 다 채워줌
        if len(rubber_col) == 1:
            single_rubber_proportion = (100 + oil_content[0])
            df_recipe[rubber_col[0]] = single_rubber_proportion

            # 잠깐 빼뒀던 마지막 폴리머 컬럼을 원래 있던 자리에 넣는 행위
            # 값: 100 - (다른 rubber phr 합)
            # 여기서 rubber column에 있는 다른애들 phr 합이 100을 넘어가는 경우가 있음 > 값이 음수가 됨
        else:
            if raw_mat_rubber_col!=[]:
                df_recipe.insert(len(rubber_col)-1, rubber_col[-1], (100 - (df_recipe[rubber_col[:-1]]/[100+i for i in oil_content[:-1]]*100).sum(axis=1)) * (100 + oil_content[-1])/100)
        


        # 고정값 받은대로 설정
        for material, value in fixed_phr_dict.items():
            if material in df_recipe.columns:
                df_recipe[material] = value
        

        # # 범위 안에 들어오는지 확인
        #     # phr max보다 rubber phr이 큰 경우 여기서 잘려나감
        #     # 고무 재료가 하나만 있고 100+oil_content가 상한을 넘어버리면, df_recipe가 비어버리는 상황이 발생
        #     # 그래서 하나만 있을때는 일단 확인안하도록 해놓음 (검토해서 수정해야할듯)


        


        # df_recipe column중에 rubber_col에 해당하는 애들 값이 음수면 해당 행 제거
        if len(rubber_col) > 1 and raw_mat_rubber_col!=[]:
            for i in rubber_col:
                df_recipe = df_recipe[df_recipe[i] >= 0]

        df_recipe[base_rm] = base_phr
        
        # AAD113A랑 AAD342A가 둘다 포함되어 있으면 AAD113A는 AAD342A의 8%로 설정해줌
        # if ('AAD113A' in df_recipe.columns) & ('AAD342A' in df_recipe.columns):
        if 'AAD342A' in df_recipe.columns:
            df_recipe['AAD113A'] = df_recipe['AAD342A']*0.08
        
        df_recipe.drop_duplicates(inplace=True)
        df_recipe.reset_index(inplace=True, drop=True)
        
        df_recipe = df_recipe.astype(float)

        # df_recipe.to_csv("recipe.csv", index=False) # 레시피 저장

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

        elapsted_time = time.time() - start
        hours, remainder = divmod(elapsted_time,3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = ""
        if hours > 0:
            time_str += f"{int(hours)} hr "
        if minutes > 0:
            time_str += f"{int(minutes)} min "
        time_str += f"{round(seconds, 2)} sec"
        st.success(f'Step 1/3 completed. Time taken: {time_str}')
        start = time.time()


        pred_g2 = loaded_model_g2.predict(df_recipe_filled)
        pred_g2_df = pd.DataFrame(pred_g2, columns=model_output_columns_g2)
        # pred_g2_df.to_csv("pred_g2.csv", index=False)

        pred_tand = pred_g2 / pred_g1
        pred_tand_df = pd.DataFrame(pred_tand, columns=['tand_-30', 'tand_0', 'tand_25', 'tand_60'])

        elapsted_time = time.time() - start
        hours, remainder = divmod(elapsted_time,3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = ""
        if hours > 0:
            time_str += f"{int(hours)} hr "
        if minutes > 0:
            time_str += f"{int(minutes)} min "
        time_str += f"{round(seconds, 2)} sec"
        st.success(f'Step 2/3 completed. Time taken: {time_str}')
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

        idx = []
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
        idx.append(0) # 레퍼런스 레시피 추가
        # print(std)
        # print(f'idx: {idx}')
        # print(len(idx))
        # print(len(set(idx)))
        # print('elapsed time, Ver 1:', round((time.time() - start), 4))

        # print(df_G1.columns)

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

        df_rank_scaling = reorder_columns(df_rank_scaling)

        # print('elapsed time, Ver 2:', round((time.time() - start), 4))

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

        df_composite_rank = reorder_columns(df_composite_rank)

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
        # df = pd.read_csv(f'{test_input_file_name}') # recipe.csv
        # df['Plasticizer']=df[['AAP501A','AAT231A']].sum(axis=1)+df['AAE325A']*0.2
        # df['Pla_Sil_Ratio'] = df['AAD342A']/df['Plasticizer']
        # df['Tg_calc'] = (df['AAE325A']*0.8*(-50)+df['AAQ233A']*(-92)+48*df['AAT231A']-101*df['AAP501A'])/df[['AAE325A','AAQ233A','AAT231A','AAP501A']].sum(axis=1)
        # df['Wet Index_New'] = (df['AAD342A']**1.8)/((df['Tg_calc'].abs())**0.8)

        input = pd.concat([df_recipe_filled_0rmv, df_G1, df_G2, df_tand], axis=1)

        # input.iloc[df_tand.index].to_csv('target_recipe_total_2_test.csv')

        target_recipe = input.iloc[idx]
        target_recipe.insert(0, 'Rank', range(1,target_recipe.shape[0]+1))
        # print('target recipe:', target_recipe)
        # target_recipe.loc[:,'rank'] = range(1,target_recipe.shape[0]+1)
        # target_recipe = target_recipe.iloc[:,:-12]

        target_recipe = reorder_columns(target_recipe)

        # target_recipe.to_csv('target_recipe_2_test.csv')

        elapsted_time = time.time() - start
        hours, remainder = divmod(elapsted_time,3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = ""
        if hours > 0:
            time_str += f"{int(hours)} hr "
        if minutes > 0:
            time_str += f"{int(minutes)} min "
        time_str += f"{round(seconds, 2)} sec"

        st.success(f'Step 3/3 completed. Time taken: {time_str}')

        total_elapsted_time = time.time() - real_start
        hours, remainder = divmod(total_elapsted_time,3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = ""
        if hours > 0:
            time_str += f"{int(hours)} hr "
        if minutes > 0:
            time_str += f"{int(minutes)} min "
        time_str += f"{round(seconds, 2)} sec"

        st.success(f'Recipe creation complete. Total time taken: {time_str}')

    ### 결과 출력
        x = st.expander('Recipe Information', expanded=True)

        x.caption('Index 0 refers to the reference recipe.')
        x.caption('The phr values for polymers displayed incorporate both the polymer and its inherent oil content.')

        x.subheader('All recipes')
        x.write('{} recipes were created'.format(df_recipe_filled_0rmv.shape[0]))
        df_recipe_filled_concat = pd.concat([df_recipe_filled_0rmv, pred_g1_df, pred_g2_df, pred_tand_df], axis=1)
        df_recipe_filled_concat = reorder_columns(df_recipe_filled_concat)
        x.dataframe(df_recipe_filled_concat, width = 800)

        x.subheader('Recipe Ranking')
        x.write('Method 1. Original Algorithm')
        x.dataframe(target_recipe, width = 800)

        x.write('Method 2. Min-max Scaling')
        x.dataframe(df_rank_scaling, width = 800)

        x.write('Method 3. Composite Ranking')
        x.dataframe(df_composite_rank, width = 800)
        


        # 기존 버전

        
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
        # print('reference:', df_recipe_filled_concat.loc[0,:])

with tab_2:
    tab_2_1, tab_2_2 = st.tabs(['Kor.', 'Eng.'])
    with tab_2_1:
        st.success('1. Recipe Range Setting')
        st.write('1-1. 가상 레시피에 들어갈 원료를 선택하고 범위를 설정합니다.')
        st.image('img/svcd_guideline_img1.png')
        st.write('')
        st.write('1-2. 범위를 몇분할 할지 나타내는 Step을 설정합니다. Step이 클수록 더 많은 레시피가 생성됩니다.')
        st.image('img/svcd_guideline_img2.png')
        st.warning('⚠️ Step 설정시 주의사항')
        st.write('- Step은 2 이상의 값으로 설정해주셔야 합니다.')
        st.write('- Step을 키울수록 레시피 생성에 소요되는 시간이 늘어나기에, 필요에 따라 적절한 값으로 설정해 주시는 것을 권장드립니다.')
        st.write('범위 설정에 들어간 원료 개수 별 적정 Step 수는 아래 표와 같습니다.')
        df_recommended_step = pd.DataFrame()
        df_recommended_step['# of materials in range setting'] = [3,4,5,6,7]
        df_recommended_step['Recommended Step range'] = ['100 - 400', '20 - 50', '10 - 20', '6 - 12', '4 - 8']
        df_recommended_step['Estimated processing time'] = ['5 sec - 5 min', '3 sec - 5 min','4 sec -  4 min', '3 sec -  4 min', '3 sec - 9 min']
        df_recommended_step['Recipes created'] = [f'{100**2+1} - {400**2+1}', f'{20**3+1} - {50**3+1}', f'{10**4+1} - {20**4+1}', f'{6**5+1} - {12**5+1}',
                                                   f'{4**6+1} - {8**6+1}']

        st.table(df_recommended_step)
        st.write('')
        st.success('2. Fixed Material Setting')
        st.write('고정값으로 들어갈 원료를 선택하고 값을 입력합니다.')
        st.image('img/svcd_guideline_img3.png')
        st.write('')
        st.success('3. Reference Recipe Setting')
        st.write('레퍼런스로 사용할 레시피를 입력합니다.')
        st.write('레퍼런스 레시피 입력 시, 추천 레시피에는 레퍼런스보다 성능이 좋을 것으로 예측되는 레시피만 표시됩니다.')
        st.image('img/svcd_guideline_img4.png')      

    with tab_2_2:
        st.success('1. Recipe Range Setting')
        st.write('1-1. Select the raw materials to be included in the virtual recipe and set the range.')
        st.image('img/svcd_guideline_img1.png')
        st.write('')
        st.write('1-2. Set the "Step", which indicates how many values the range will be splitted into. The larger the "Step", the more recipes generated.')
        st.image('img/svcd_guideline_img2.png')
        st.warning('⚠️ Notes on setting up the "Step"')
        st.write('- "Step" must be set to a value of 2 or higher.')
        st.write('- Increasing the "Step" will increase the time it takes to generate the recipe, so we recommend that you set it to an appropriate value based on your needs.')
        st.write('The table below shows the appropriate number of "Step" for each number of ingredients in the range setting.')
        st.table(df_recommended_step)
        st.write('')
        st.success('2. Fixed Material Setting')
        st.write('Select the raw materials that would take fixed phr and enter the values.')
        st.image('img/svcd_guideline_img3.png')
        st.write('')
        st.success('3. Reference Recipe Setting')
        st.write('Enter a recipe to use as a reference.')
        st.write('When a reference recipe is entered, only those recipes that are predicted to perform better than the reference are shown in the recommended recipes.')
        st.image('img/svcd_guideline_img4.png')

with tab_3:
    st.write('[2024.04.25]')
    st.write('- Application released')


with tab_4:
    with st.form(key='my_form'):
        text_input = st.text_area(label='Please give any suggestions about the application.')
        submit_button = st.form_submit_button(label='Submit')

    def save_variable_to_file(variable, directory="."):
        """
        Saves a variable to a text file named suggestion_{number}.txt within the specified directory.
        The number is determined based on existing files to avoid overwriting.
        
        Args:
        variable (str): The content to save in the file.
        directory (str): The directory to check and save the file in.
        """
        # List all files in the directory
        files = os.listdir(directory)
        
        # Pattern to capture numbers in files starting with 'suggestion'
        pattern = r'^suggestion_(\d+)\.txt$'
        
        # Find the highest current number used in filenames
        max_number = 0
        for file in files:
            match = re.match(pattern, file)
            if match:
                current_number = int(match.group(1))
                if current_number > max_number:
                    max_number = current_number
        
        # Determine the new file number
        new_number = max_number + 1
        
        # Create a new filename
        new_filename = f"suggestion_{new_number}.txt"
        
        # Path for the new file
        filepath = os.path.join(directory, new_filename)
        
        # Writing the variable to the file
        with open(filepath, 'w') as f:
            f.write(str(variable))
        
        return new_filename  # Optional: return the new filename for confirmation

    if submit_button:
        saved_filename = save_variable_to_file(text_input, 'suggestions')
    
