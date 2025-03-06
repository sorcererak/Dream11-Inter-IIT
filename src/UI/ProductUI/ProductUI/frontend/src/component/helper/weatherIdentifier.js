const dataMap = {
  sunny: {
    temp_range: [18, 30],
    humidity_range: [30, 60],
    effect: "High scores, and spin-friendly.",
    image_url: "/weather/sunny.png",
  },
  overcast: {
    temp_range: [15, 25],
    humidity_range: [60, 85],
    effect: "Swing bowlers get help, low scores possible.",
    image_url: "/weather/overcast.png",
  },
  rainy: {
    temp_range: [10, 22],
    humidity_range: [70, 100],
    effect: "Disrupted matches, DLS method applied.",
    image_url: "/weather/rainy.png",
  },
  humid: {
    temp_range: [18, 35],
    humidity_range: [60, 100],
    effect: "Extra swing for bowlers, affects fitness.",
    image_url: "/weather/humid.png",
  },
  windy: {
    temp_range: [10, 30],
    humidity_range: [30, 70],
    effect: "Ball movement in the air; tough for batsmen.",
    image_url: "/weather/windy.png",
  },
  cold: {
    temp_range: [-10, 5],
    humidity_range: [40, 80],
    effect: "Players may struggle with grip and stamina.",
    image_url: "/weather/cold.png",
  },
  hot_dry: {
    temp_range: [30, 45],
    humidity_range: [30, 60],
    effect: "Favor spinners, high-scoring games likely.",
    image_url: "/weather/hot_dry.png",
  },
  stormy: {
    temp_range: [18, 35],
    humidity_range: [60, 100],
    effect: "Interruptions, spin dominates wet pitches.",
    image_url: "/weather/stormy.png",
  },
  hazy: {
    temp_range: [20, 35],
    humidity_range: [40, 70],
    effect: "Dew impacts evening matches; slippery balls.",
    image_url: "/weather/hazy.png",
  },
  foggy: {
    temp_range: [0, 15],
    humidity_range: [90, 100],
    effect: "Reduced visibility; challenging for fielding.",
    image_url: "/weather/foggy.png",
  },
};

const inRange = (w_type, temp, humid) => {
  return (
    w_type.humidity_range[0] <= humid &&
    humid <= w_type.humidity_range[1] &&
    w_type.temp_range[0] <= temp &&
    temp <= w_type.temp_range[1]
  );
};

const WhichWeather = (temp, humid) => {
  for (let w_type in dataMap) {
    if (inRange(dataMap[w_type], temp, humid)) {
      return {
        effect: dataMap[w_type].effect,
        name: w_type,
        url: dataMap[w_type].image_url,
      };
    }
  }
  return {
    effect: "Ideal conditions for all players; balanced play",
    name: "Clear/Mild",
    url: "/weather/sun.png",
  };
};

export default WhichWeather;
