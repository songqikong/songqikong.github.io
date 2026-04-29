---
layout: page
full-width: true
title: "Political Test"
---

<div class="oc-layout" style="height: calc(100vh - 50px); overflow: auto;">
  <!-- Top Bar -->
  <div class="oc-topbar oc-fade-in">
    <div class="oc-topbar-left">
      <span class="oc-prompt">$</span>
      <span class="oc-cmd">./political_test</span>
      <span style="color: var(--oc-text-tertiary);">// 8values compass</span>
    </div>
    <div class="oc-topbar-right">
      <div class="oc-status-dot"></div>
      <span class="oc-status-text">running</span>
    </div>
  </div>

  <!-- Content -->
  <div class="pt-content oc-fade-in oc-delay-1">
    <div class="pt-header">
      <span class="pt-logo">&gt;_political_test</span>
      <span class="pt-ver">v1.0</span>
      <span class="pt-subtitle">64 questions, 4 axes, 32 identity labels — for entertainment only</span>
    </div>

    <div class="pt-toolbar">
      <div class="pt-progress-bar"><span id="ptProgress"></span></div>
      <div class="pt-count" id="ptCount">0 / 64</div>
      <button class="pt-btn pt-btn-primary" id="ptSubmit" disabled>查看结果</button>
    </div>

    <div id="ptQuestions" class="pt-questions"></div>

    <aside id="ptRoastToast" class="pt-roast-toast" aria-live="polite">
      <span id="ptRoastBadge" class="pt-roast-badge">Q1</span>
      <span id="ptRoastText" class="pt-roast-text"></span>
      <button id="ptRoastClose" class="pt-roast-close" type="button" aria-label="关闭吐槽">×</button>
    </aside>

    <section id="ptResult" class="pt-result">
      <h2>测试结果</h2>
      <div class="pt-grid" id="ptAxisResults"></div>
      <h2>身份标签匹配</h2>
      <div class="pt-matches" id="ptMatches"></div>
      <p class="pt-note">标签采用四维坐标近邻匹配，不代表固定身份归类；越接近只表示本次答题坐标与该标签预设坐标更相似。</p>
    </section>
  </div>

  <!-- Bottom Bar -->
  <div class="oc-bottombar oc-fade-in oc-delay-2">
    <div class="oc-stats">
      <div class="oc-stat">
        <span class="oc-stat-value">64</span>
        <span class="oc-stat-label">questions</span>
      </div>
      <div class="oc-stat">
        <span class="oc-stat-value">4</span>
        <span class="oc-stat-label">axes</span>
      </div>
    </div>
    <div style="font-family: var(--oc-mono); font-size: 11px; color: var(--oc-text-tertiary);">
      ~/vibing/political_test <span class="oc-cursor"></span>
    </div>
  </div>
</div>

<style>
.pt-content {
  padding: 12px 16px;
  max-width: 860px;
  margin: 0 auto;
}

.pt-header {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.pt-logo {
  font-family: var(--oc-mono);
  font-size: 18px;
  font-weight: 700;
  color: var(--oc-blue);
}

.pt-ver {
  font-family: var(--oc-mono);
  font-size: 11px;
  color: var(--oc-text-tertiary);
  padding: 2px 6px;
  border: 1px solid var(--oc-border);
  border-radius: 4px;
}

.pt-subtitle {
  font-size: 13px;
  color: var(--oc-text-secondary);
  margin-left: 4px;
}

.pt-toolbar {
  display: grid;
  grid-template-columns: 1fr auto auto;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  margin-bottom: 12px;
  background: var(--oc-bg-alt);
  border: 1px solid var(--oc-border);
  border-radius: var(--oc-radius);
  position: sticky;
  top: 0;
  z-index: 5;
}

.pt-progress-bar {
  height: 8px;
  background: var(--oc-bg-card);
  border-radius: 999px;
  overflow: hidden;
}

.pt-progress-bar > span {
  display: block;
  width: 0%;
  height: 100%;
  background: var(--oc-blue);
  border-radius: 999px;
  transition: width 160ms ease;
}

.pt-count {
  color: var(--oc-text-tertiary);
  font-family: var(--oc-mono);
  font-size: 13px;
  white-space: nowrap;
}

.pt-btn {
  border: 1px solid var(--oc-border);
  border-radius: 6px;
  background: var(--oc-bg-alt);
  color: var(--oc-text);
  min-height: 34px;
  padding: 0 14px;
  font: inherit;
  font-size: 13px;
  cursor: pointer;
  transition: all var(--oc-transition);
}

.pt-btn-primary {
  border-color: var(--oc-blue);
  background: var(--oc-blue);
  color: #fff;
  font-weight: 600;
}

.pt-btn-primary:hover:not(:disabled) {
  background: var(--oc-blue-light);
  border-color: var(--oc-blue-light);
}

.pt-btn:disabled {
  cursor: not-allowed;
  opacity: 0.4;
}

.pt-questions {
  display: grid;
  gap: 10px;
}

.pt-question {
  padding: 14px 16px;
  background: var(--oc-bg-alt);
  border: 1px solid var(--oc-border);
  border-radius: var(--oc-radius);
  transition: border-color var(--oc-transition);
}

.pt-question:hover {
  border-color: var(--oc-blue-border);
}

.pt-q-head {
  display: grid;
  grid-template-columns: 34px 1fr;
  gap: 10px;
  align-items: start;
  margin-bottom: 10px;
}

.pt-qid {
  min-width: 34px;
  color: var(--oc-text-tertiary);
  font-family: var(--oc-mono);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}

.pt-qtext {
  font-size: 14px;
  line-height: 1.6;
  color: var(--oc-text);
}

.pt-choices {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 6px;
}

.pt-choice {
  position: relative;
}

.pt-choice input {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}

.pt-choice span {
  display: grid;
  place-items: center;
  min-height: 34px;
  padding: 5px 6px;
  border: 1px solid var(--oc-border);
  border-radius: 6px;
  color: var(--oc-text-secondary);
  text-align: center;
  font-size: 12px;
  user-select: none;
  cursor: pointer;
  transition: all var(--oc-transition);
  background: var(--oc-bg-card);
}

.pt-choice span:hover {
  border-color: var(--oc-blue-border);
  color: var(--oc-blue);
}

.pt-choice input:checked + span {
  border-color: var(--oc-blue);
  background: var(--oc-blue-bg);
  color: var(--oc-blue);
  font-weight: 700;
}

/* Roast Toast */
.pt-roast-toast {
  position: fixed;
  left: 50%;
  bottom: 18px;
  z-index: 20;
  width: min(700px, calc(100% - 28px));
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 10px;
  align-items: start;
  padding: 12px 14px;
  border: 1px solid var(--oc-blue-border);
  border-radius: 8px;
  background: var(--oc-bg-terminal);
  box-shadow: 0 18px 48px rgba(0,0,0,0.5);
  backdrop-filter: blur(10px);
  transform: translate(-50%, calc(100% + 28px));
  opacity: 0;
  pointer-events: none;
  transition: transform 160ms ease, opacity 160ms ease;
}

.pt-roast-toast.show {
  transform: translate(-50%, 0);
  opacity: 1;
  pointer-events: auto;
}

.pt-roast-badge {
  display: inline-grid;
  place-items: center;
  min-width: 38px;
  height: 26px;
  padding: 0 8px;
  border-radius: 999px;
  background: var(--oc-blue);
  color: #fff;
  font-weight: 800;
  font-size: 12px;
  font-family: var(--oc-mono);
  font-variant-numeric: tabular-nums;
}

.pt-roast-text {
  color: #C9D1D9;
  font-size: 13px;
  line-height: 1.55;
}

.pt-roast-close {
  width: 26px;
  min-height: 26px;
  padding: 0;
  border: 0;
  border-radius: 50%;
  background: var(--oc-bg-card);
  color: var(--oc-text-secondary);
  font-size: 18px;
  line-height: 1;
  cursor: pointer;
}

.pt-roast-close:hover {
  background: var(--oc-border);
  color: var(--oc-text);
}

/* Result */
.pt-result {
  display: none;
  margin-top: 18px;
}

.pt-result.show {
  display: block;
}

.pt-result h2 {
  font-size: 18px;
  margin: 18px 0 10px;
  color: var(--oc-text);
  font-weight: 700;
}

.pt-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.pt-axis-panel {
  padding: 14px 16px;
  background: var(--oc-bg-alt);
  border: 1px solid var(--oc-border);
  border-radius: var(--oc-radius);
}

.pt-score-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  font-weight: 700;
  font-size: 14px;
  color: var(--oc-text);
}

.pt-score-title b {
  color: var(--oc-blue);
  font-variant-numeric: tabular-nums;
  font-size: 16px;
}

.pt-axis-row {
  display: grid;
  grid-template-columns: 50px 1fr 50px;
  gap: 8px;
  align-items: center;
  margin: 8px 0 0;
  color: var(--oc-text-tertiary);
  font-size: 12px;
}

.pt-bar {
  position: relative;
  height: 12px;
  background: var(--oc-bg-card);
  border-radius: 999px;
  overflow: hidden;
}

.pt-bar span {
  display: block;
  height: 100%;
  width: 50%;
  background: linear-gradient(90deg, #f87171, #fbbf24, #34d399);
  border-radius: 999px;
  transition: width 0.4s ease;
}

.pt-matches {
  display: grid;
  gap: 10px;
  margin-top: 10px;
}

.pt-match-hint {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  width: fit-content;
  padding: 6px 10px;
  border: 1px solid var(--oc-blue-border);
  border-radius: 999px;
  background: var(--oc-blue-bg);
  color: var(--oc-blue);
  font-size: 12px;
  font-weight: 700;
}

.pt-top-match {
  display: grid;
  grid-template-columns: 80px 1fr auto;
  gap: 14px;
  align-items: start;
  padding: 16px;
  border: 1px solid var(--oc-blue);
  border-radius: var(--oc-radius);
  background: var(--oc-blue-bg);
}

.pt-top-match svg {
  width: 80px;
  height: 80px;
  display: block;
}

.pt-top-match h3 {
  margin: 0;
  font-size: 22px;
  letter-spacing: 0;
  color: var(--oc-text);
}

.pt-top-match p {
  margin: 6px 0 0;
  color: var(--oc-text-secondary);
  font-size: 13px;
  line-height: 1.55;
}

.pt-top-match small {
  color: var(--oc-text-tertiary);
  font-size: 11px;
}

.pt-top-score {
  min-width: 64px;
  color: var(--oc-blue);
  font-size: 24px;
  font-weight: 800;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.pt-match {
  display: grid;
  grid-template-columns: 42px 1fr auto;
  gap: 12px;
  align-items: center;
  padding: 10px 12px;
  border: 1px solid var(--oc-border);
  border-radius: 6px;
  background: var(--oc-bg-alt);
  cursor: pointer;
  text-align: left;
  width: 100%;
  font: inherit;
  color: var(--oc-text);
  transition: all var(--oc-transition);
}

.pt-match:hover,
.pt-match:focus-visible {
  border-color: var(--oc-blue);
  background: var(--oc-blue-bg);
  outline: none;
}

.pt-match.active {
  border-color: var(--oc-blue);
  box-shadow: inset 3px 0 0 var(--oc-blue);
}

.pt-match svg {
  width: 42px;
  height: 42px;
  display: block;
}

.pt-match strong {
  font-size: 14px;
  display: block;
}

.pt-match small {
  display: block;
  color: var(--oc-text-tertiary);
  margin-top: 2px;
  font-size: 11px;
}

.pt-match b {
  color: var(--oc-blue);
  font-size: 16px;
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}

.pt-click-chip {
  display: inline-block;
  margin-top: 3px;
  color: var(--oc-blue);
  font-size: 11px;
  font-weight: 700;
}

.pt-note {
  color: var(--oc-text-tertiary);
  font-size: 12px;
  margin: 12px 0 0;
  line-height: 1.6;
}

@media (max-width: 700px) {
  .pt-content { padding: 8px 10px; }
  .pt-choices { grid-template-columns: 1fr; }
  .pt-grid { grid-template-columns: 1fr; }
  .pt-top-match { grid-template-columns: 60px 1fr; }
  .pt-top-match svg { width: 60px; height: 60px; }
  .pt-top-score { grid-column: 1 / -1; text-align: left; font-size: 20px; }
  .pt-toolbar { grid-template-columns: 1fr; }
}
</style>

<script>
(function() {
  const scale = [
    ["强烈赞同", 2],
    ["赞同", 1],
    ["中立", 0],
    ["反对", -1],
    ["强烈反对", -2]
  ];

  const axisMeta = {
    economic: ["市场", "平等", "经济"],
    diplomatic: ["国际", "民族", "外交"],
    civil: ["权威", "自由", "民权"],
    societal: ["传统", "进步", "社会"]
  };

  const labelProfiles = {
    "马列毛派": [95, 65, 35, 55],
    "安那其": [95, 25, 95, 80],
    "社民": [80, 35, 75, 75],
    "加速主义者": [75, 45, 60, 55],
    "工业党": [65, 75, 30, 45],
    "神友": [55, 10, 70, 65],
    "劳权卫士": [90, 45, 70, 65],
    "精神资本家": [10, 45, 55, 35],
    "皇汉": [45, 95, 35, 20],
    "小粉红": [55, 90, 30, 45],
    "入关学徒": [50, 95, 25, 30],
    "离岸爱国者": [45, 85, 40, 45],
    "逆向民族主义者": [45, 5, 60, 65],
    "世界公民": [55, 5, 80, 85],
    "润学实践者": [40, 10, 75, 65],
    "精外": [45, 15, 55, 50],
    "建制派": [50, 75, 15, 40],
    "实装人大派": [55, 55, 65, 50],
    "自由派知识分子": [45, 25, 90, 70],
    "法家信徒": [45, 75, 5, 20],
    "技术利维坦派": [55, 65, 10, 45],
    "公民社会倡导者": [60, 30, 90, 80],
    "深蓝建制": [45, 75, 20, 35],
    "赛博朋克反抗者": [55, 20, 95, 75],
    "激进女权": [70, 45, 70, 95],
    "进步主义者": [65, 25, 80, 95],
    "老派儒教徒": [40, 70, 25, 5],
    "田园保守派": [35, 65, 30, 10],
    "LGBT 权益者": [55, 25, 80, 95],
    "国学爱好者": [45, 75, 45, 15],
    "新保守主义者": [35, 65, 35, 15],
    "解构大圣": [50, 30, 75, 70]
  };

  const labelDescriptions = {
    "马列毛派": "来自中文互联网长期存在的左翼亚文化，在贫富分化、劳资冲突和怀旧叙事中持续复活。它代表一种强平等、反资本、重组织和重阶级分析的立场，常把当代问题解释为生产关系失衡。",
    "安那其": "多见于反权威、反资本和亚文化圈层，受无政府主义、自治社群和去中心化想象影响。它代表对国家、资本和等级秩序的同时怀疑，强调个体自治、互助网络和自发秩序。",
    "社民": "形成于自由派与网左之间的温和改良地带，常从北欧福利国家、劳工保护和公共服务汲取语言。它代表在市场框架内扩大再分配、福利保障和程序政治的路线。",
    "加速主义者": "兴起于对温和改良失望后的网络犬儒与激进想象，常把危机看成逼出变化的燃料。它代表一种认为系统矛盾需要被推到极限、让旧秩序自行暴露问题的立场。",
    "工业党": "来自工程师、制造业和发展主义论坛传统，反感空谈价值，迷信生产力、产业链和技术官僚。它代表以国家能力、技术升级和工业效率为核心的现代化叙事。",
    "神友": "源自贴吧、抽象社区和失败叙事的混合体，在自嘲、黑色幽默和反宏大叙事中成型。它代表低期待、强解构和对现实生存压力的阴阳怪气式表达。",
    "劳权卫士": "由 996、平台用工、外卖骑手和裁员潮等议题催生，关注具体劳动处境而非纯理论站队。它代表对劳动保护、集体协商、反过劳和基本尊严的直接诉求。",
    "精神资本家": "常见于商业评论区、创业神话和成功学语境，把自己代入资本与管理者视角。它代表市场竞争、私有产权、效率优先和对福利再分配的本能警惕。",
    "皇汉": "在民族身份焦虑、历史叙事和文化复兴浪潮中扩大影响，强调汉文化主体性。它代表强民族认同、文化边界意识和对多元叙事的排斥或警惕。",
    "小粉红": "成长于社交媒体民族主义、国际摩擦和国家成就叙事之中，情绪动员能力强。它代表维护国家形象、反外部批评和对共同体荣誉的高度敏感。",
    "入关学徒": "来自对国际秩序不平等的现实主义解读，把世界体系理解为丛林规则。它代表中国必须增强实力、重塑规则、从被动适应转向主动进场的想象。",
    "离岸爱国者": "多见于留学、移民和跨境生活群体中的身份延续，空间在外但叙事仍向内。它代表对本国国家叙事的持续认同，也常被争议为立场与生活选择之间的张力。",
    "逆向民族主义者": "在启蒙叙事、社会批判和失望情绪中形成，常把本土问题归结为深层文化缺陷。它代表对民族自豪叙事的反向抵抗，但也容易滑向简单化的自我否定。",
    "世界公民": "来自自由主义、国际主义和跨国生活经验，强调普遍权利与全球责任。它代表淡化民族边界、重视人权、气候、移民和全球协作的价值取向。",
    "润学实践者": "由就业焦虑、教育竞争、公共风险和跨国机会共同塑造，是个人主义迁徙策略的网络化命名。它代表把个人前途和生活质量置于国家叙事之前的选择。",
    "精外": "由对特定外国制度、历史阶段、军政美学或生活方式的迷恋组成，常带二次元和亚文化色彩。它代表把外部对象当作审美、秩序或政治想象的投射屏幕。",
    "建制派": "来自秩序优先、发展成就和国家能力叙事，倾向认为稳定是改革与生活的前提。它代表对集中治理、强领导和现有制度连续性的信任。",
    "实装人大派": "形成于程序改良派和制度内参与想象，强调不另起炉灶，而是把既有规则真正跑起来。它代表信息公开、预算监督、听证、人大和行政诉讼等制度化路径。",
    "自由派知识分子": "继承 80、90 年代启蒙话语，关注个人权利、言论空间、法治和宪政程序。它代表反权力任性、重公共讨论和以普遍价值审视本土现实的立场。",
    "法家信徒": "由秩序焦虑、治安想象和高效治理崇拜催生，常把严刑峻法视为社会稳定工具。它代表强制、纪律、中央权威和对混乱的低容忍。",
    "技术利维坦派": "兴起于大数据、AI、实名制和智慧城市叙事，把技术治理视作解决社会复杂性的钥匙。它代表用监控、算法和平台化管理换取安全、效率与可控性。",
    "公民社会倡导者": "来自 NGO、志愿者、社区自治和公共议题行动传统，重视国家与市场之外的中间空间。它代表结社、互助、地方治理和公共参与的扩张。",
    "深蓝建制": "是建制派中的精英主义版本，反感群众运动和情绪政治，偏好专业官僚和稳定秩序。它代表以秩序、技术能力和精英管理压制民粹冲动的倾向。",
    "赛博朋克反抗者": "由平台监控、数据泄露、实名制和算法支配焦虑催生，常与加密、去中心化和隐私工具相连。它代表对数字权力的敏感和以技术反制技术的冲动。",
    "激进女权": "在婚育压力、职场歧视、性别暴力和网络对线中变得尖锐，强调结构性父权批判。它代表强烈的性别政治意识、反规训和对温和改良的不耐烦。",
    "进步主义者": "由女权、LGBTQ、残障、反歧视和多元文化议题汇合而成，常被保守派称作 woke。它代表对少数群体处境、身份平权和包容性制度的支持。",
    "老派儒教徒": "源自传统伦理复兴、家族秩序想象和对现代个人主义的不满。它代表等级、礼法、亲亲尊尊和以传统道德修补现代社会裂缝的愿望。",
    "田园保守派": "常在乡土想象、反智情绪和朴素道德判断中出现，把新观念视为外来麻烦。它代表经验主义、传统家庭观和对现代平权议题的防御性反感。",
    "LGBT 权益者": "由性少数社群、公共卫生、反歧视和亲密关系合法化议题推动形成。它代表对性倾向和性别身份平等、反污名和法律承认的持续诉求。",
    "国学爱好者": "在文化消费、传统审美和民族复兴叙事中壮大，常从汉服、典籍、礼仪和生活方式切入。它代表文化守旧与审美复古，有时温和，有时滑向道德规训。",
    "新保守主义者": "由生育焦虑、家庭解体、性别冲突和价值失序感推动，强调家庭、责任与道德共识。它代表反极端平权、反解构和以传统核心制度维持社会稳定。",
    "解构大圣": "来自 B 站、贴吧、抽象话和二次元模因文化，擅长把严肃叙事做成梗。它代表对宏大叙事和身份站队的消解，用玩笑、二创和阴阳怪气保持距离。"
  };

  const questions = [
    ["平台经济的效率不应建立在外卖骑手、网约车司机和客服人员缺乏劳动保障的基础上。", [10, 0, 0, 0]],
    ["高收入者和大型企业应承担更高税负，用于教育、医疗、住房等公共服务。", [10, 0, 0, 0]],
    ["只要不违法，资本追求利润最大化就是社会发展的主要动力，不应被过度道德化批评。", [-10, 0, 0, 0]],
    ["关键基础设施、自然资源和基础民生行业更适合由公共部门或强监管机构掌控。", [10, 0, 0, 0]],
    ["福利国家容易养懒人，社会应更多奖励自我奋斗和市场竞争中的胜出者。", [-10, 0, 0, 0]],
    ["比起纯粹的市场价格信号，产业政策和技术官僚规划更能解决长期发展问题。", [7, 0, -3, 0]],
    ["年轻人的低欲望、摆烂和自嘲，更多是结构性压力的反映，而不是个人道德失败。", [8, 0, 2, 0]],
    ["大厂裁员、降薪或延长工时是企业在竞争中自救的正常选择，外部不应过多干预。", [-10, 0, 0, 0]],
    ["私有产权和企业家精神比再分配政策更能创造长期繁荣。", [-10, 0, 0, 0]],
    ["住房不应主要被视为投资品，政府应强力压低住房金融属性。", [9, 0, 0, 1]],
    ["经济不平等只要来自能力差异和合法竞争，就不必被视为严重社会问题。", [-10, 0, 0, 0]],
    ["工会、行业协会或劳动者集体协商机制应在企业治理中拥有更实际的权力。", [10, 0, 3, 0]],
    ["与其修修补补，不如让垄断资本和不合理制度的矛盾充分暴露，逼出根本变化。", [7, 0, 2, 0]],
    ["在国际争议中，维护本国形象和国家利益通常应优先于讨好外部舆论。", [0, 10, 0, 0]],
    ["民族身份是政治共同体的重要基础，不能被抽象的世界主义轻易取代。", [0, 10, 0, -2]],
    ["个人选择移民、留学或跨国工作不应被赋予道德背叛的含义。", [0, -8, 3, 0]],
    ["中国应更多参与并改造国际秩序，而不是只适应既有西方规则。", [0, 8, 0, 0]],
    ["评价一个社会时，普遍人权和个体尊严应高于民族自尊叙事。", [0, -9, 5, 2]],
    ["传统汉文化在现代国家认同中应拥有更核心的位置。", [0, 8, 0, -5]],
    ["过度强调本民族的缺陷，往往只会复制另一种简单化的民族主义。", [0, 7, 0, 0]],
    ["历史和现实中的外国制度经验值得优先学习，本土特殊性经常被用来拒绝改革。", [0, -8, 2, 1]],
    ["身在海外仍积极维护中国立场是正常的身份延续，而不是矛盾或虚伪。", [0, 8, 0, 0]],
    ["气候、公共卫生和难民等问题需要超越国界的合作，国家利益不应总是压倒全球责任。", [0, -10, 1, 2]],
    ["国际关系本质上是强者制定规则，弱者谈道德只会被继续收割。", [0, 9, -1, 0]],
    ["对某个外国历史阶段或意识形态的迷恋，不应替代对现实本土问题的判断。", [0, 6, 0, 0]],
    ["言论自由应保护刺耳、冒犯和反主流的表达，除非它直接煽动现实伤害。", [0, 0, 10, 2]],
    ["为了社会稳定，网络平台和公共舆论场需要更严格、更主动的管理。", [0, 0, -10, 0]],
    ["完善人大、听证、信息公开和行政诉讼等既有程序，比街头动员更可靠。", [0, 0, 5, 0]],
    ["在重大危机中，集中决策和强制执行比程序讨论更重要。", [0, 0, -9, 0]],
    ["严刑峻法虽然看似冷酷，但能有效压低犯罪和社会失序成本。", [0, 0, -10, -2]],
    ["大数据、实名制和智能监控只要能提升治理效率，就值得扩大使用。", [0, 0, -10, 0]],
    ["隐私保护和加密技术是普通人抵抗滥权的重要工具。", [0, 0, 10, 1]],
    ["公益组织、志愿者网络和社区自治应拥有更大空间参与社会治理。", [2, 0, 9, 2]],
    ["群众运动往往会走向非理性，因此公共事务最好由受过专业训练的精英处理。", [0, 0, -8, -1]],
    ["国家权力越强，就越需要可验证的法律边界和独立监督。", [0, 0, 10, 1]],
    ["普通人不必过多参与政治，只要生活水平持续提高，强有力的治理就是可接受的。", [0, 0, -10, 0]],
    ["政治参与不应只停留在情绪表达，还应落实到公开预算、问责和程序性权利。", [0, 0, 9, 1]],
    ["如果一种技术治理能显著减少欺诈、犯罪和谣言，牺牲部分匿名空间是合理的。", [0, 0, -9, 0]],
    ["性别不平等不是个别男性或女性的问题，而是需要制度和文化共同改变的结构性问题。", [1, 0, 1, 10]],
    ["LGBTQ+ 群体应拥有平等的婚姻、收养和反歧视保护。", [0, 0, 2, 10]],
    ["家庭稳定和传统性别分工是社会秩序的重要支柱，不宜被轻易解构。", [0, 0, 0, -10]],
    ["国学、宗族和传统礼仪可以提供现代社会缺失的道德共同体。", [0, 3, 0, -9]],
    ["网络女权对父权制的尖锐批评，即使让人不舒服，也有其现实基础。", [1, 0, 1, 10]],
    ["过度强调少数群体权益会撕裂社会共识，并损害多数人的正常生活。", [0, 0, -1, -10]],
    ["残障者、性少数、外来务工者等群体的处境，应成为公共政策的核心考量。", [2, 0, 1, 10]],
    ["年轻人对婚恋、生育和家庭责任的抵触，说明现代个人主义已经过度膨胀。", [0, 0, -1, -9]],
    ["社会对女性外貌、年龄和婚育状态的审判，比对男性更严苛，应被持续挑战。", [0, 0, 1, 10]],
    ["学校和媒体应避免过早引入性别多元、性少数等议题，以免冲击未成年人的价值观。", [0, 0, -1, -10]],
    ["用模因和讽刺消解宏大叙事，有助于打破道德绑架和身份站队。", [0, -1, 4, 5]],
    ["一个社会需要稳定的道德共识，不能把所有传统规范都视为压迫。", [0, 1, -1, -10]],
    ["互联网平台的算法推荐和流量分配应接受公共监督，因为它们已经影响就业、舆论和文化生产。", [6, 0, 5, 2]],
    ["创业者和投资人承担了巨大风险，因此获得远高于普通劳动者的收益是合理的。", [-9, 0, 0, 0]],
    ["教育、医疗和养老的基本保障应尽量去商品化，不能主要依赖个人购买能力。", [10, 0, 0, 2]],
    ["国家产业竞争中的技术自主，比短期消费自由和进口便利更重要。", [2, 8, -2, 0]],
    ["批评本国政策或社会问题不等于不爱国，公共讨论应允许这种区分。", [0, -5, 8, 2]],
    ["面对外部压力时，内部争议应暂时让位于一致对外。", [0, 8, -6, -1]],
    ["全球化带来的人员流动和文化混合总体上丰富了社会，而不是削弱共同体。", [0, -9, 3, 6]],
    ["公共安全部门在调查犯罪时，应更容易获得个人通信和平台数据。", [0, 0, -9, -1]],
    ["即使某些公共抗议会造成不便，它们仍是社会表达利益和纠错的重要方式。", [2, -1, 10, 4]],
    ["司法和执法的公开透明比快速结案更能维护长期秩序。", [0, 0, 9, 1]],
    ["生育、婚姻和家庭责任应更多被视为个人选择，而不是对国家或宗族的义务。", [0, -2, 5, 9]],
    ["流行文化中的性别反串、酷儿表达和亚文化身份没有必要被过度警惕。", [0, -1, 4, 10]],
    ["传统节日、礼仪和家族记忆应被积极传承，否则社会会失去根基。", [0, 4, -1, -9]],
    ["对冒犯性言论的社会抵制是合理的，但不应轻易发展成封号、开除或现实围剿。", [0, 0, 8, 4]]
  ];

  const axisKeys = ["economic", "diplomatic", "civil", "societal"];
  const answers = new Array(questions.length).fill(null);

  function renderQuestions() {
    const root = document.getElementById("ptQuestions");
    root.innerHTML = questions.map((q, index) =>
      '<div class="pt-question">' +
        '<div class="pt-q-head">' +
          '<div class="pt-qid">' + String(index + 1).padStart(2, "0") + '</div>' +
          '<div class="pt-qtext">' + q[0] + '</div>' +
        '</div>' +
        '<div class="pt-choices">' +
          scale.map(function(s) {
            return '<label class="pt-choice">' +
              '<input type="radio" name="pq' + index + '" value="' + s[1] + '" />' +
              '<span>' + s[0] + '</span>' +
            '</label>';
          }).join("") +
        '</div>' +
      '</div>'
    ).join("");

    root.addEventListener("change", function(event) {
      var input = event.target;
      if (!input.matches("input[type='radio']")) return;
      var index = Number(input.name.slice(2));
      answers[index] = Number(input.value);
      renderRoast(index, answers[index]);
      updateProgress();
    });
  }

  function dominantAxis(effects) {
    var axisIndex = 0, max = 0;
    effects.forEach(function(effect, index) {
      if (Math.abs(effect) > max) { axisIndex = index; max = Math.abs(effect); }
    });
    return axisIndex;
  }

  function stanceFor(index, answerValue) {
    if (answerValue === 0) return "neutral";
    var effects = questions[index][1];
    var axisIndex = dominantAxis(effects);
    var signed = effects[axisIndex] * answerValue;
    var axis = axisKeys[axisIndex];
    var side = signed > 0 ? "positive" : "negative";
    return axis + ":" + side;
  }

  function intensity(answerValue) { return Math.abs(answerValue) === 2 ? "hard" : "soft"; }

  function pick(list, seed) { return list[Math.abs(seed) % list.length]; }

  function renderRoast(index, answerValue) {
    var hard = intensity(answerValue) === "hard";
    var neutral = [
      "中立是吧，典型岁静端水怪，锅都烧穿了你还在旁边装饮水机。",
      "这题你选择隐身，属于小红书式“我不站队但我都懂”，懂完等于没懂。",
      "不表态也是一种表态：吧友鉴定为滑跪预备役，谁赢你就给谁递纸巾。",
      "你这波主打不粘锅人格，油都溅脸上了还在问能不能各退一步。",
      "好一个理中客，左看右看都不亏，成分检测仪扫完显示“废话浓度超标”。"
    ];
    var roasts = {
      "economic:positive": [
        hard ? "好家伙，工友大喇叭直接焊手上了，老板放个屁你都能听出剩余价值的臭味。" : "你这边明显往打工人抱团去了，安卓人互助会今日新增一位嘴替。",
        hard ? "资本家看了沉默，HR 看了装死，你这劳权浓度高到能把绩效表泡烂。" : "你这不是单纯反资本，是被绩效表扇醒后终于不替老板数钱了。",
        hard ? "工贼雷达反向爆表，吧友鉴定：你已经进入“万恶的资本”复读机模式。" : "有点网左味了，但还没到把奶茶店吸管都批判成生产资料的程度。",
        hard ? "小红书打工人看了直接收藏，标题：被公司当耗材后，我终于不装孙子了。" : "你这发言属于“我不是想闹，我只是想别被当电池榨干”。",
        hard ? "这题你像刚从 996 福报垃圾堆里爬出来，张嘴就是资本家别装人。" : "你这不是安卓人抱团罢了，是饼吃多了终于发现老板连水都不给。"
      ],
      "economic:negative": [
        hard ? "工贼指数拉满，老板还没张嘴，你已经跪着把降本增效四个字擦亮了。" : "你这精神股东味儿上来了，股份没拿到，替资本洗地倒是先转正了。",
        hard ? "好一位赛博监工，KPI 还没发你就开始抽同事鞭子，HR 看了都嫌你卷。" : "你这套话术像从商业区热评臭水沟里捞出来的，主打市场包治百病。",
        hard ? "资本家缺的不是律师，是你这种免费护城河，纯爱战士都没你对老板纯。" : "有点精神资本家，但还保留了下班后偷骂老板两句的卑微弹性。",
        hard ? "吧友锐评：你不是支持市场，你是工位坐久了坐出董事长幻觉，醒醒，椅子还是公司的。" : "小红书标题：普通人如何拥有老板思维。评论区：先拥有老板收入，别光拥有老板嘴脸。",
        hard ? "这题属于福报派大弟子，工资可以没有，格局必须通天，PUA 闭环都给你盘活了。" : "你这边偏资本叙事，嘴上说效率，心里已经把同事打包进优化名单。"
      ],
      "diplomatic:positive": [
        hard ? "民族叙事油门踩穿了，键盘边境线都被你磨出火星，热评区缺你这个护旗手。" : "你这已经进入爱国评论区热身区，差一口气就能把置顶热评盘包浆。",
        hard ? "入关味儿很浓，世界地图在你眼里只剩战略纵深，连厕所门口都想画势力范围。" : "国家队滤镜开得不低，但还没到见人先查户口本成分的程度。",
        hard ? "皇汉/小粉红雷达同时爆响，属于旗帜先举，逻辑后补，反驳先扣帽三件套。" : "你这题明显更信共同体叙事，个人小算盘滚一边，宏大叙事先上炕。",
        hard ? "吧友鉴定：赢学浓度过高，评论区已经自动生成“赢赢赢，最后赢到麻”。" : "你这发言有点赢麻了前摇，属于先把民族情绪灌满再假装分析。",
        hard ? "这题你像刚刷完三小时地缘政治短视频，开口棋局闭口大势，细节全靠脑补。" : "你这边偏民族叙事，不是不能聊世界，但得先把本方 buff 叠到冒烟。"
      ],
      "diplomatic:negative": [
        hard ? "世界公民浓度爆表，边界感被你踹进垃圾桶，护照都快被供成精神牌位。" : "你这润学窗口已经弹出来了，只差点确认并重启人生。",
        hard ? "逆民和普世派正在抢你头像框，系统提示：民族叙事被你卸载得连缓存都没了。" : "你更吃国际主义这套，民族叙事在你这儿得先排队挨骂。",
        hard ? "河殇味儿都飘出来了，评论区马上问你是不是跪久了腿麻还说地球很圆。" : "你这不是不爱本土，是对宏大口号过敏，听见口号就想反胃。",
        hard ? "小红书定位已切海外生活区，标题：我为什么拒绝被集体叙事绑架。评论：姐你太清醒。" : "你这边有点世界公民滤镜，吧友看了会说你外宾发言别太满。",
        hard ? "这题你像在评论区手搓民间版联合国，国界线看了都想辞职。" : "你对民族叙事不太上头，主打先做人，再看身份证是不是烫手。"
      ],
      "civil:positive": [
        hard ? "自由派雷达爆表，实名制看了你都想给自己打码，摄像头当场装死。" : "你这明显偏公民社会，遇事先问程序正义有没有签到，没签到就开喷。",
        hard ? "赛博反抗者上号了，隐私设置被你翻到冒烟，平台后台看了直呼这人真烦。" : "你对权力边界挺敏感，看到管理两个字就跟被踩尾巴一样弹起来。",
        hard ? "公民权利浓度过高，建制评论区马上给你扣一顶“不稳定因素体验卡”，还包邮。" : "你这套回答很程序派，先别急着热血，证据链甩脸上再说。",
        hard ? "贴吧老哥锐评：你这不是提意见，你是想让权力写周报，写完还得被你批注。" : "你这边偏自由主义，最烦的不是被管，是被管了还不让问凭什么，憋屈得想骂街。",
        hard ? "这题你像开了去中心化滤镜，凡是集中管理都得先被你扒层皮验成分。" : "你对公权力有天然防沉迷系统，强制弹窗你都想点举报加拉黑。"
      ],
      "civil:negative": [
        hard ? "法家信徒坐实了，建议把评论区也纳入郡县制管理，杠精统一拉去填表。" : "你这秩序优先味儿挺明显，先管起来，理由后补，票都不用买。",
        hard ? "技术利维坦看了直呼内行，你恨不得给社会装后台，一键清理所有“不懂事缓存”。" : "你对强治理接受度不低，属于宁可误伤一片，也别让场面难看。",
        hard ? "建制浓度很足，群众一开麦你就想收麦克风，主打一个大家都闭嘴世界就好了。" : "你这题偏权威，安全感来自整齐划一，最好连表情包都统一编号。",
        hard ? "吧友鉴定：你不是喜欢秩序，你是想把世界做成 Excel，谁越格谁挨骂。" : "你这属于管理学上头，看到混乱就想喊管理员封楼顺便禁言。",
        hard ? "这题你像评论区秩序保安，先别吵，全体排队领系统提示，谁不服谁异常。" : "你这边偏建制，公共讨论在你眼里像没关提示音的群聊，烦得想全员禁言。"
      ],
      "societal:positive": [
        hard ? "进步浓度拉满，老保评论区血压爆表，家族群已经准备把你踢了再说你不孝。" : "你这明显偏平权派，看到传统规训就想拆开看看里面是不是全是爹味零件。",
        hard ? "女权/LGBT 友好度爆表，催婚话术当场报废，七大姑八大姨集体沉默三秒。" : "你对少数群体权益挺敏感，属于看见歧视就开麦，麦克风还带混响。",
        hard ? "Woke 雷达亮成夜店灯球，保守派看了连夜写小作文：现在的年轻人真完了。" : "你这题偏进步，但还没到把所有饭局都改成议题批斗会。",
        hard ? "小红书集美看了直呼姐妹清醒，评论区已经开始互递电子榔头，谁爹味砸谁。" : "你这边进步味儿很足，主打谁规训我，我就给谁做一套文本解剖。",
        hard ? "这题你像把父权制拖进评论区公开处刑，顺手给传统观念打一星差评：爹味太冲。" : "你这不是单纯支持多元，是对爹味、规训、凝视三件套生理性反胃。"
      ],
      "societal:negative": [
        hard ? "老保味儿太冲了，祖宗牌位都快被你搬进评论区当核武器，弹幕全是家风家训。" : "你这传统滤镜开着，现代观念进门前得先给你磕个头。",
        hard ? "新保守主义直接上桌，家庭价值四个字在你这儿像免死金牌，谁反驳谁不懂事。" : "你更信稳定伦理，看到解构两个字就想给它一巴掌让它坐好。",
        hard ? "国学复兴小组欢迎你，下一步是不是给年轻人安排族谱 KPI 和孝道 OKR，完不成扣祖德。" : "你这题明显偏传统，秩序感比自我表达更让你有安全感，最好大家都归位。",
        hard ? "吧友鉴定：爹味含量超标，建议开窗；这套话术发家族群能拿五个大拇指和一个红包。" : "你这边老派价值观上头，看到新概念第一反应是“什么玩意也敢上桌”。",
        hard ? "这题你像小红书传统婚恋顾问，开口稳定闭口责任，主打把所有人按回旧格子里。" : "你这不是单纯保守，是觉得社会最好像老式衣柜，谁乱放谁欠收拾。"
      ]
    };
    var stance = stanceFor(index, answerValue);
    var text = stance === "neutral" ? pick(neutral, index) : pick(roasts[stance], index + answerValue * 7);
    document.getElementById("ptRoastBadge").textContent = "Q" + (index + 1);
    document.getElementById("ptRoastText").textContent = text;
    document.getElementById("ptRoastToast").classList.add("show");
  }

  function updateProgress() {
    var done = answers.filter(function(v) { return v !== null; }).length;
    document.getElementById("ptCount").textContent = done + " / " + questions.length;
    document.getElementById("ptProgress").style.width = (done / questions.length * 100) + "%";
    document.getElementById("ptSubmit").disabled = done !== questions.length;
  }

  function calculateScores() {
    var raw = [0, 0, 0, 0], max = [0, 0, 0, 0];
    questions.forEach(function(q, index) {
      q[1].forEach(function(effect, axisIndex) {
        raw[axisIndex] += answers[index] * effect;
        max[axisIndex] += 2 * Math.abs(effect);
      });
    });
    return raw.map(function(value, index) {
      return Math.round((value + max[index]) / (2 * max[index]) * 100);
    });
  }

  function axisLabel(score, low, high) {
    if (score >= 80) return "强" + high;
    if (score >= 60) return "偏" + high;
    if (score > 40) return "中间";
    if (score > 20) return "偏" + low;
    return "强" + low;
  }

  function similarity(distance) {
    var maxDistance = 200;
    return Math.max(0, Math.round((1 - distance / maxDistance) * 100));
  }

  function hashLabel(label) {
    return [].reduce.call(label, function(sum, ch, i) { return sum + ch.charCodeAt(0) * (i + 7); }, 0);
  }

  function makeIcon(label, profile, size) {
    size = size || 80;
    var hash = hashLabel(label);
    var colors = {
      "马列毛派": ["#b91c1c", "#7f1d1d"], "安那其": ["#111827", "#dc2626"],
      "社民": ["#e11d48", "#be123c"], "加速主义者": ["#7c3aed", "#111827"],
      "工业党": ["#475569", "#0f766e"], "神友": ["#64748b", "#111827"],
      "劳权卫士": ["#dc2626", "#f97316"], "精神资本家": ["#16a34a", "#0f766e"],
      "皇汉": ["#b45309", "#7c2d12"], "小粉红": ["#ec4899", "#dc2626"],
      "入关学徒": ["#991b1b", "#1f2937"], "离岸爱国者": ["#2563eb", "#dc2626"],
      "逆向民族主义者": ["#334155", "#7f1d1d"], "世界公民": ["#0ea5e9", "#10b981"],
      "润学实践者": ["#0284c7", "#f59e0b"], "精外": ["#6366f1", "#0f172a"],
      "建制派": ["#1d4ed8", "#1e3a8a"], "实装人大派": ["#d97706", "#b45309"],
      "自由派知识分子": ["#2563eb", "#7c3aed"], "法家信徒": ["#111827", "#b45309"],
      "技术利维坦派": ["#0891b2", "#1e293b"], "公民社会倡导者": ["#059669", "#2563eb"],
      "深蓝建制": ["#1e3a8a", "#0f172a"], "赛博朋克反抗者": ["#a855f7", "#06b6d4"],
      "激进女权": ["#db2777", "#7e22ce"], "进步主义者": ["#14b8a6", "#8b5cf6"],
      "老派儒教徒": ["#92400e", "#78350f"], "田园保守派": ["#65a30d", "#854d0e"],
      "LGBT 权益者": ["#ef4444", "#8b5cf6"], "国学爱好者": ["#b45309", "#dc2626"],
      "新保守主义者": ["#475569", "#92400e"], "解构大圣": ["#f97316", "#7c3aed"]
    };
    var c = colors[label] || ["hsl(" + (hash % 360) + " 70% 46%)", "hsl(" + ((hash + 80) % 360) + " 70% 34%)"];
    var symbols = {
      "马列毛派": '<path d="M48 18l7.5 18.5 20 1.5-15.2 12.8 4.8 19.5L48 60 30.9 70.3l4.8-19.5L20.5 38l20-1.5z" fill="#fde68a"/><path d="M31 63c11-19 24-29 39-31" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round"/><path d="M34 62c9-4 20-5 31 0" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round"/>',
      "安那其": '<circle cx="48" cy="48" r="28" fill="none" stroke="#fff" stroke-width="6"/><path d="M30 72l18-48 18 48M38 53h20" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/><path d="M14 82L82 14" stroke="#ef4444" stroke-width="8" stroke-linecap="round"/>',
      "社民": '<path d="M48 58c-13-3-20-11-18-21 2-11 15-13 22-5 8-8 21-4 20 8-1 12-12 17-24 18z" fill="#fecdd3"/><path d="M47 31c4 7 2 17-4 24M51 32c-4 8-1 17 5 22M34 42c9 0 18 5 27 0" fill="none" stroke="#be123c" stroke-width="3" stroke-linecap="round"/><path d="M48 57c-6 7-10 12-23 15 12 6 26-1 31-12" fill="#16a34a"/><path d="M50 58c8 1 14 4 21 12-11 2-20-2-25-10" fill="#22c55e"/>',
      "加速主义者": '<path d="M24 55h18L33 78l31-38H46l11-22z" fill="#facc15" stroke="#fff" stroke-width="3" stroke-linejoin="round"/><path d="M18 28c20-12 42-7 57 9M20 40c18-9 34-6 48 6" fill="none" stroke="#fff" stroke-width="4" stroke-linecap="round" opacity=".75"/>',
      "工业党": '<circle cx="48" cy="48" r="15" fill="none" stroke="#fff" stroke-width="7"/><path d="M48 15v12M48 69v12M15 48h12M69 48h12M25 25l9 9M62 62l9 9M71 25l-9 9M34 62l-9 9" stroke="#fff" stroke-width="7" stroke-linecap="round"/><path d="M25 73h46" stroke="#fbbf24" stroke-width="6" stroke-linecap="round"/>',
      "神友": '<path d="M28 34c4-9 17-13 27-6 13-5 24 5 20 19-4 17-24 27-48 26 7-5 10-10 9-16-11-4-13-14-8-23z" fill="#e2e8f0"/><circle cx="42" cy="45" r="4" fill="#111827"/><circle cx="61" cy="45" r="4" fill="#111827"/><path d="M42 59c5 4 13 4 19 0" fill="none" stroke="#111827" stroke-width="4" stroke-linecap="round"/><path d="M23 27l10 6M72 30l9-7" stroke="#fff" stroke-width="4" stroke-linecap="round"/>',
      "劳权卫士": '<path d="M34 44c0-10 4-16 9-16 2 0 4 2 4 5v13h2V29c0-4 7-4 7 0v17h2V34c0-4 7-4 7 0v17c0 17-8 25-22 25-11 0-19-8-19-20V44c0-5 10-5 10 0z" fill="#fee2e2" stroke="#7f1d1d" stroke-width="4" stroke-linejoin="round"/><path d="M24 75h48" stroke="#fff" stroke-width="5" stroke-linecap="round"/>',
      "精神资本家": '<path d="M26 70h44c4 0 7-3 7-7V39c0-4-3-7-7-7H26c-4 0-7 3-7 7v24c0 4 3 7 7 7z" fill="#dcfce7" stroke="#14532d" stroke-width="4"/><path d="M37 32v-7h22v7M48 41v18M39 48c0-5 4-8 9-8s9 3 9 7c0 10-18 3-18 12 0 4 4 7 9 7s9-3 9-8" fill="none" stroke="#16a34a" stroke-width="4" stroke-linecap="round"/>',
      "皇汉": '<path d="M48 17l23 13v19c0 14-8 25-23 31-15-6-23-17-23-31V30z" fill="#fef3c7" stroke="#92400e" stroke-width="4"/><path d="M35 37h26M39 48h18M33 60h30" stroke="#92400e" stroke-width="5" stroke-linecap="round"/><path d="M48 25v47" stroke="#dc2626" stroke-width="4" stroke-linecap="round"/>',
      "小粉红": '<path d="M25 50c-7-15 10-28 23-16 13-12 30 1 23 16-5 12-23 23-23 23S30 62 25 50z" fill="#fbcfe8" stroke="#be185d" stroke-width="4"/><path d="M48 28l4 9 10 .8-7.6 6.3 2.4 9.9L48 48.8 39.2 54l2.4-9.9L34 37.8l10-.8z" fill="#fde047"/>',
      "入关学徒": '<path d="M24 74h48V36H24z" fill="#e5e7eb" stroke="#111827" stroke-width="4"/><path d="M30 36c4-12 12-18 18-18s14 6 18 18" fill="none" stroke="#111827" stroke-width="5"/><path d="M20 56h33M53 56l-11-10M53 56L42 66" stroke="#dc2626" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>',
      "离岸爱国者": '<circle cx="48" cy="48" r="30" fill="#dbeafe" stroke="#1d4ed8" stroke-width="4"/><path d="M23 48h50M48 18c-10 10-10 50 0 60M48 18c10 10 10 50 0 60" fill="none" stroke="#1d4ed8" stroke-width="3"/><path d="M48 30l4 9 10 1-8 6 3 10-9-5-9 5 3-10-8-6 10-1z" fill="#ef4444"/>',
      "逆向民族主义者": '<path d="M26 28h44v40H26z" fill="#e2e8f0" stroke="#111827" stroke-width="4"/><path d="M32 34l32 28M64 34L32 62" stroke="#7f1d1d" stroke-width="6" stroke-linecap="round"/><path d="M25 75c14-7 31-7 46 0" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round"/>',
      "世界公民": '<circle cx="48" cy="48" r="31" fill="#d1fae5" stroke="#0284c7" stroke-width="4"/><path d="M21 48h54M48 17c-12 12-12 50 0 62M48 17c12 12 12 50 0 62M27 33c14 5 28 5 42 0M27 63c14-5 28-5 42 0" fill="none" stroke="#0284c7" stroke-width="3"/><path d="M48 33l5 10 11 2-8 8 2 11-10-5-10 5 2-11-8-8 11-2z" fill="#22c55e"/>',
      "润学实践者": '<rect x="25" y="38" width="42" height="31" rx="5" fill="#fef3c7" stroke="#92400e" stroke-width="4"/><path d="M36 38v-8h20v8M67 54h14M81 54l-8-8M81 54l-8 8" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/><circle cx="35" cy="72" r="4" fill="#92400e"/><circle cx="58" cy="72" r="4" fill="#92400e"/>',
      "精外": '<path d="M48 19l8 17 19 2-14 13 4 19-17-9-17 9 4-19-14-13 19-2z" fill="#c7d2fe" stroke="#312e81" stroke-width="4"/><path d="M31 75h34" stroke="#fff" stroke-width="5" stroke-linecap="round"/><circle cx="48" cy="48" r="9" fill="#fff"/>',
      "建制派": '<path d="M24 72h48M29 66V39h38v27M25 39h46L48 22z" fill="#dbeafe" stroke="#1e3a8a" stroke-width="4" stroke-linejoin="round"/><path d="M38 45v16M48 45v16M58 45v16" stroke="#1e3a8a" stroke-width="4" stroke-linecap="round"/>',
      "实装人大派": '<rect x="25" y="22" width="46" height="58" rx="5" fill="#fef3c7" stroke="#92400e" stroke-width="4"/><path d="M35 36h26M35 48h26M35 60h15" stroke="#92400e" stroke-width="4" stroke-linecap="round"/><path d="M56 65l6 6 13-17" fill="none" stroke="#16a34a" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>',
      "自由派知识分子": '<path d="M24 29c10-7 18-6 24 0 6-6 14-7 24 0v41c-10-5-18-5-24 1-6-6-14-6-24-1z" fill="#dbeafe" stroke="#1d4ed8" stroke-width="4" stroke-linejoin="round"/><path d="M48 29v42M33 40h9M33 51h9M54 40h9M54 51h9" stroke="#1d4ed8" stroke-width="3" stroke-linecap="round"/>',
      "法家信徒": '<path d="M48 18v55M29 35h38M34 35l-12 25h24zM62 35L50 60h24z" fill="none" stroke="#fbbf24" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/><path d="M32 76h32" stroke="#fff" stroke-width="5" stroke-linecap="round"/>',
      "技术利维坦派": '<rect x="26" y="26" width="44" height="44" rx="8" fill="#cffafe" stroke="#0e7490" stroke-width="4"/><path d="M37 26v-9M48 26v-9M59 26v-9M37 79v-9M48 79v-9M59 79v-9M26 37h-9M26 48h-9M26 59h-9M79 37h-9M79 48h-9M79 59h-9" stroke="#fff" stroke-width="4" stroke-linecap="round"/><circle cx="48" cy="48" r="10" fill="#0e7490"/><circle cx="48" cy="48" r="4" fill="#fff"/>',
      "公民社会倡导者": '<circle cx="33" cy="39" r="8" fill="#dcfce7" stroke="#166534" stroke-width="4"/><circle cx="63" cy="39" r="8" fill="#dbeafe" stroke="#1d4ed8" stroke-width="4"/><circle cx="48" cy="63" r="8" fill="#fef3c7" stroke="#92400e" stroke-width="4"/><path d="M40 43l16 0M37 47l7 10M59 47l-7 10" stroke="#fff" stroke-width="5" stroke-linecap="round"/>',
      "深蓝建制": '<path d="M48 18l24 10v18c0 16-9 27-24 34-15-7-24-18-24-34V28z" fill="#bfdbfe" stroke="#172554" stroke-width="4"/><path d="M33 45h30M38 56h20M48 30v38" stroke="#172554" stroke-width="5" stroke-linecap="round"/>',
      "赛博朋克反抗者": '<rect x="30" y="43" width="36" height="30" rx="5" fill="#e9d5ff" stroke="#6b21a8" stroke-width="4"/><path d="M37 43v-9c0-8 5-13 11-13s11 5 11 13v9" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round"/><path d="M20 32h12M64 32h12M20 64h10M66 64h10M48 73v10" stroke="#22d3ee" stroke-width="4" stroke-linecap="round"/><circle cx="48" cy="58" r="5" fill="#6b21a8"/>',
      "激进女权": '<circle cx="48" cy="36" r="18" fill="#fce7f3" stroke="#be185d" stroke-width="5"/><path d="M48 54v25M36 67h24M58 26l11-11M69 15v13M69 15H56" stroke="#be185d" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>',
      "进步主义者": '<path d="M22 60c12-22 28-31 52-32-7 23-20 38-46 42 10-9 19-17 29-27-15 7-25 16-35 17z" fill="#ccfbf1" stroke="#0f766e" stroke-width="4" stroke-linejoin="round"/><path d="M25 74h45" stroke="#a78bfa" stroke-width="6" stroke-linecap="round"/>',
      "老派儒教徒": '<path d="M26 70h44V33H26z" fill="#fef3c7" stroke="#78350f" stroke-width="4"/><path d="M22 33h52L48 18zM33 70V42M48 70V42M63 70V42" fill="none" stroke="#78350f" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><path d="M37 55h22" stroke="#dc2626" stroke-width="4" stroke-linecap="round"/>',
      "田园保守派": '<path d="M22 63c10-22 26-35 52-40-4 28-19 45-47 50 9-11 19-22 31-33-15 7-27 15-36 23z" fill="#bef264" stroke="#3f6212" stroke-width="4"/><path d="M27 76h43" stroke="#854d0e" stroke-width="6" stroke-linecap="round"/>',
      "LGBT 权益者": '<path d="M22 30h52v36H22z" fill="#fff" stroke="#1f2937" stroke-width="4"/><path d="M25 35h46" stroke="#ef4444" stroke-width="6"/><path d="M25 42h46" stroke="#f97316" stroke-width="6"/><path d="M25 49h46" stroke="#eab308" stroke-width="6"/><path d="M25 56h46" stroke="#22c55e" stroke-width="6"/><path d="M25 63h46" stroke="#3b82f6" stroke-width="6"/><path d="M48 73c-10-8-18-13-18-22 0-10 12-12 18-4 6-8 18-6 18 4 0 9-8 14-18 22z" fill="#fff" opacity=".9"/>',
      "国学爱好者": '<path d="M29 24h38v48H29z" fill="#fef3c7" stroke="#92400e" stroke-width="4"/><path d="M35 31h26M35 41h26M35 51h20" stroke="#92400e" stroke-width="4" stroke-linecap="round"/><path d="M24 72h48M48 24v48" stroke="#dc2626" stroke-width="4" stroke-linecap="round"/>',
      "新保守主义者": '<path d="M23 45l25-21 25 21v29H23z" fill="#e2e8f0" stroke="#334155" stroke-width="4" stroke-linejoin="round"/><path d="M38 74V55h20v19M36 45h24" stroke="#92400e" stroke-width="5" stroke-linecap="round"/><path d="M48 32l7 8-7 8-7-8z" fill="#fbbf24"/>',
      "解构大圣": '<path d="M25 58c10-14 18-22 35-31 4 17 0 31-15 43 2-10 5-18 8-26-9 7-16 14-28 14z" fill="#fed7aa" stroke="#9a3412" stroke-width="4"/><path d="M32 35c-2-10 3-17 13-18M60 29c8-5 15-3 20 5M36 68c10 6 22 5 32-4" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round"/><circle cx="43" cy="47" r="4" fill="#111827"/><circle cx="58" cy="43" r="4" fill="#111827"/>'
    };
    var symbol = symbols[label] || '<text x="48" y="56" text-anchor="middle" font-size="22" font-weight="800" fill="#fff" font-family="Microsoft YaHei, sans-serif">' + label.slice(0, 2) + '</text>';
    return '<svg viewBox="0 0 96 96" width="' + size + '" height="' + size + '" role="img" aria-label="' + label + ' 图标">' +
      '<defs><linearGradient id="g' + hash + '" x1="0" y1="0" x2="1" y2="1">' +
        '<stop offset="0%" stop-color="' + c[0] + '" />' +
        '<stop offset="100%" stop-color="' + c[1] + '" />' +
      '</linearGradient></defs>' +
      '<rect x="8" y="8" width="80" height="80" rx="18" fill="url(#g' + hash + ')" />' +
      '<circle cx="48" cy="48" r="36" fill="rgba(255,255,255,0.12)" />' +
      symbol +
    '</svg>';
  }

  function renderResult() {
    var scores = calculateScores();
    var axisResults = document.getElementById("ptAxisResults");
    axisResults.innerHTML = axisKeys.map(function(key, index) {
      var meta = axisMeta[key];
      var score = scores[index];
      return '<div class="pt-axis-panel">' +
        '<div class="pt-score-title">' +
          '<span>' + meta[2] + '轴：' + axisLabel(score, meta[0], meta[1]) + '</span>' +
          '<b>' + score + '</b>' +
        '</div>' +
        '<div class="pt-axis-row">' +
          '<span>' + meta[0] + '</span>' +
          '<div class="pt-bar"><span style="width:' + score + '%"></span></div>' +
          '<span>' + meta[1] + '</span>' +
        '</div>' +
      '</div>';
    }).join("");

    var matches = Object.entries(labelProfiles)
      .map(function(entry) {
        var label = entry[0], profile = entry[1];
        var distance = Math.sqrt(profile.reduce(function(sum, value, index) {
          return sum + Math.pow(scores[index] - value, 2);
        }, 0));
        return {
          label: label,
          distance: distance,
          similarity: similarity(distance),
          profile: profile,
          description: labelDescriptions[label] || "这个标签暂时没有描述。"
        };
      })
      .sort(function(a, b) { return a.distance - b.distance; })
      .slice(0, 6);

    var top = matches[0];
    document.getElementById("ptMatches").innerHTML =
      '<div class="pt-match-hint">点开下面任意身份标签，可以查看它的完整描述</div>' +
      '<div class="pt-top-match" id="ptMatchDetail"></div>' +
      matches.map(function(item, index) {
        return '<button class="pt-match ' + (index === 0 ? "active" : "") + '" type="button" data-match-index="' + index + '">' +
          makeIcon(item.label, item.profile, 42) +
          '<div>' +
            '<strong>' + (index + 1) + '. ' + item.label + '</strong>' +
            '<small>预设坐标：[平等 ' + item.profile[0] + '，民族 ' + item.profile[1] + '，自由 ' + item.profile[2] + '，进步 ' + item.profile[3] + ']</small>' +
            '<span class="pt-click-chip">点击查看描述</span>' +
          '</div>' +
          '<b>' + item.similarity + '%</b>' +
        '</button>';
      }).join("");

    var showMatchDetail = function(item) {
      document.getElementById("ptMatchDetail").innerHTML =
        makeIcon(item.label, item.profile, 80) +
        '<div>' +
          '<h3>' + item.label + '</h3>' +
          '<p>' + item.description + '</p>' +
          '<small>预设坐标：[平等 ' + item.profile[0] + '，民族 ' + item.profile[1] + '，自由 ' + item.profile[2] + '，进步 ' + item.profile[3] + ']</small>' +
        '</div>' +
        '<div class="pt-top-score">' + item.similarity + '%</div>';
    };
    showMatchDetail(top);

    document.querySelectorAll("[data-match-index]").forEach(function(button) {
      button.addEventListener("click", function() {
        var index = Number(button.dataset.matchIndex);
        showMatchDetail(matches[index]);
        document.querySelectorAll("[data-match-index]").forEach(function(item) { item.classList.remove("active"); });
        button.classList.add("active");
        document.getElementById("ptMatchDetail").scrollIntoView({ behavior: "smooth", block: "nearest" });
      });
    });

    document.getElementById("ptResult").classList.add("show");
    document.getElementById("ptResult").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  document.getElementById("ptSubmit").addEventListener("click", renderResult);
  document.getElementById("ptRoastClose").addEventListener("click", function() {
    document.getElementById("ptRoastToast").classList.remove("show");
  });

  renderQuestions();
  updateProgress();
})();
</script>
