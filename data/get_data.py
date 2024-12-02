import json

translated_data = [
    {"query": "다섯 명의 여성이 해변을 따라 플립플롭을 신고 걸어간다.", "pos": ["플립플롭을 신은 몇몇 여성들이 해변을 따라 걸어가고 있다"], "neg": ["4명의 여성이 해변에 앉아 있다.", "1996년에 개혁이 있었다.", "그녀는 자신의 기록을 정정하기 위해 법정에 가지 않을 것이다.", "그 남자는 하와이에 대해 이야기하고 있다.", "한 여성이 밖에 서 있다.", "전투는 끝났다.", "한 무리의 사람들이 배구를 하고 있다."]},
    {"query": "한 여성이 높은 절벽 위에서 한 발로 서서 강을 내려다보고 있다.", "pos": ["한 여성이 절벽 위에 서 있다."], "neg": ["한 여성이 의자에 앉아 있다.", "조지 부시는 공화당원들에게 최고 고문들의 조언에 반하여 이 어리석은 생각을 고려조차 하지 않겠다고 말했다.", "그 가족은 무너지고 있었다.", "아무도 회의에 나타나지 않았다", "한 소년이 밖에서 모래를 가지고 놀고 있다.", "전보를 받자마자 끝났다.", "한 아이가 자기 방에서 책을 읽고 있다."]},
    {"query": "두 여성이 악기를 연주하고 있다; 한 명은 클라리넷, 다른 한 명은 바이올린을 연주한다.", "pos": ["몇 사람이 곡을 연주하고 있다."], "neg": ["두 여성이 기타와 드럼을 연주하고 있다.", "한 남자가 산을 스키를 타고 내려가고 있다.", "살인자가 생각했던 때에 치명적인 용량이 투여되지 않았다.", "자전거를 타고 있는 사람", "그 소녀는 아치길에 기대어 서 있다.", "한 무리의 여성들이 소파 오페라를 보고 있다.", "사람들은 나이가 들어도 절대 잊지 않는다."]},
    {"query": "파란색 탱크톱을 입은 소녀가 앉아서 세 마리의 개를 지켜보고 있다.", "pos": ["한 소녀가 파란색을 입고 있다."], "neg": ["한 소녀가 세 마리의 고양이와 함께 있다.", "사람들이 장례 행렬을 지켜보고 있다.", "그 아이는 검은색을 입고 있다.", "공립학교에서 우리에게 재정은 문제이다.", "수영장에 있는 아이들.", "폭행당하는 것은 진정시키는 일이다.", "나는 18살에 심각한 문제에 직면했다."]},
    {"query": "노란 개가 숲길을 따라 달리고 있다.", "pos": ["개가 달리고 있다"], "neg": ["고양이가 달리고 있다", "스틸은 그녀의 원래 이야기를 지키지 않았다.", "이 규칙은 사람들이 자녀 양육비를 내는 것을 막는다.", "조끼를 입은 남자가 차 안에 앉아 있다.", "검은 옷을 입고 흰색 반다나와 선글라스를 낀 사람이 버스 정류장에서 기다리고 있다.", "글로브나 메일 중 어느 쪽도 캐나다의 현재 도로 체계 상태에 대해 언급하지 않았다.", "스프링 크릭 시설은 오래되고 구식이다."]},
    {"query": "각 단계에서의 필수 활동과 그 활동들과 관련된 중요한 요소들을 설명한다.", "pos": ["필수 활동에 대한 중요 요소들이 설명되어 있다."], "neg": ["중요한 활동들을 설명하지만 그 활동들과 관련된 중요한 요소들에 대한 규정은 없다.", "사람들이 항의하기 위해 모여 있다.", "주 정부는 당신이 그렇게 하기를 선호할 것이다.", "한 소녀가 한 소년 옆에 앉아 있다.", "두 남성이 공연하고 있다.", "아무도 뛰고 있지 않다", "콘라드는 머리를 맞도록 음모를 꾸미고 있었다."]},
    {"query": "한 남자가 레스토랑에서 연설을 하고 있다.", "pos": ["한 사람이 연설을 하고 있다."], "neg": ["그 남자는 테이블에 앉아 음식을 먹고 있다.", "이것은 확실히 승인이 아니다.", "그들은 은퇴 때문에 집을 팔았지, 대출 때문이 아니다.", "미주리 주의 인장은 완벽하다.", "누군가가 손을 들고 있다.", "한 운동선수가 1500미터 수영 경기에 참가하고 있다.", "두 남자가 마술 쇼를 보고 있다."]},
    {"query": "인디언들이 코트를 입고 음식과 음료를 가지고 모임을 갖고 있다.", "pos": ["인디언 그룹이 음식과 음료를 가지고 모임을 갖고 있다"], "neg": ["인디언 그룹이 장례식을 하고 있다", "이것은 팔마의 큰 투우장에서 겨울 오후에만 공연된다.", "올바른 정보는 법률 서비스 관행과 사법 체계를 강화할 수 있다.", "한편, 본토는 인구가 없었다.", "두 아이가 자고 있다.", "어부가 원숭이를 잡으려고 하고 있다", "사람들이 기차 안에 있다"]},
    {"query": "보라색 머리를 한 여성이 밖에서 자전거를 타고 있다.", "pos": ["한 여성이 자전거를 타고 있다."], "neg": ["한 여성이 공원에서 조깅을 하고 있다.", "그 거리는 하얀색으로 칠해진 집들로 가득했다.", "한 그룹이 안에서 영화를 보고 있다.", "소풍에서 남자들이 스테이크를 자르고 있다", "여러 명의 요리사들이 앉아서 음식에 대해 이야기하고 있다.", "위원회는 중요한 대안들이 고려되지 않았다고 지적한다.", "우리는 장작이 다 떨어져서 불을 위해 소나무 바늘을 사용해야 했다."]},
    {"query": "한 남자가 도시 거리에서 인력거로 두 여성을 끌고 있다.", "pos": ["한 남자가 도시에 있다."], "neg": ["한 남자가 비행기 조종사이다.", "그것은 지루하고 평범하다.", "아침 햇살이 밝게 비치고 따뜻했다.", "두 사람이 부두에서 뛰어내렸다.", "사람들이 우주선 발사를 보고 있다.", "테레사 수녀는 쉬운 선택이다.", "원하는 속도로 갈 수 있는 것은 가치가 있다."]}
]

with open('toy_finetune_data.jsonl', 'w', encoding='utf-8') as f:
    for item in translated_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')