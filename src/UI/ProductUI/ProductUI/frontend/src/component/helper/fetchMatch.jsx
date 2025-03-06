const MatchDataFetch = async (url, setFunction) => {
  try {
    const response = await fetch(url, {
      method: "GET",
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(
        `HTTP Error: ${response.status} - ${response.statusText}`
      );
    }
    const dist = data.data.filter(
      (obj, index, self) =>
        index === self.findIndex((t) => t.match_id === obj.match_id)
    );
    setFunction(dist);
    console.log(dist);
  } catch (error) {
    // alert("We encountered an issue. Please try again later.");
    console.error("Error fetching teams:", error);
  }
};

export default MatchDataFetch;
