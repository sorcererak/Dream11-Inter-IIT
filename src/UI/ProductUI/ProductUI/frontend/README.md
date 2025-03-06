# Dream11 Fantasy Cricket Platform

A robust fantasy cricket platform powered by AI for team generation and analysis. The platform leverages machine learning models and real-time data to provide intelligent team suggestions and match analysis.

## Features

### AI-Generated Dream Teams
- **ML-Based Selection**: Utilizes historical player data and current conditions
- **Performance Prediction**: Advanced algorithms for player performance forecasting
- **Team Optimization**: Balanced team composition based on multiple factors
- **Real-time Updates**: Dynamic team adjustments based on match conditions

### Custom Match Creation
- **CSV Data Import**: Bulk player data import functionality
- **Manual Selection**: Interactive player selection interface
- **Team Validation**: Automatic team composition validation
- **Custom Rules**: Configurable team formation rules

### Player Analysis & Statistics
- **Radar Charts**: Visual representation of player statistics
- **Performance Metrics**: Comprehensive player performance tracking
- **Historical Data**: Past performance analysis and trends
- **Comparison Tools**: Player-to-player statistical comparison

### Weather Integration
- **Real-time Weather Data**: Live weather condition monitoring
- **Impact Analysis**: Weather effects on player performance
- **Condition-based Optimization**: Team adjustments based on weather
- **Historical Weather Patterns**: Analysis of performance in similar conditions

## Technical Architecture

### Frontend Architecture
- **React 18.3**: Latest React features including concurrent rendering
- **Component Structure**: Modular design with reusable components
- **State Management**: React Context for global state
- **Routing**: React Router v7 for navigation
- **Code Splitting**: Lazy loading for optimal performance

### UI Libraries & Integration
- **Material UI**: Core UI components and theming
- **Tailwind CSS**: Utility-first styling approach
- **Material Tailwind**: Enhanced Material Design components
- **PrimeReact**: Advanced UI components
- **Chart.js**: Data visualization
- **React DnD**: Drag and drop functionality
- **React Joyride**: Interactive user tours

### Data Flow
```
User Input → React Components → Context API → Data Processing → UI Update
```

## Project Structure

```
src/
├── assets/          # Static assets and images
│   ├── landing_page/  # Landing page assets
│   └── weather/       # Weather icons and images
├── ChatBot/         # AI chatbot implementation
│   ├── ChatBot1.jsx   # Main chatbot component
│   └── ChatBubble.jsx # Message bubble component
├── component/
│   ├── charts/      # Data visualization
│   │   └── radarChart.jsx  # Player stats radar chart
│   ├── common/      # Reusable components
│   │   ├── dreamPoints.jsx # Points display
│   │   └── weatherCard.jsx # Weather information
│   ├── helper/      # Utility functions
│   │   ├── cleanPlayerType.jsx # Player type normalization
│   │   ├── fetchMatch.jsx     # Match data fetching
│   │   └── weatherIdentifier.js # Weather condition logic
│   └── StarterPage/ # Landing page components
├── css/            # Stylesheets
└── pages/          # Main page components
```

## Implementation Details

### Weather System Implementation
```javascript
const weatherTypes = {
  sunny: {
    temp_range: [18, 30],
    humidity_range: [30, 60],
    effect: "High scores, spin-friendly"
  },
  // ... other weather types
}
```

### Player Performance Calculation
```javascript
// Dream Points Calculation
const DreamPointsCard = ({ points }) => (
  <div className="bg-transparent">
    <div className="flex items-center">
      <LuCoins />
      <p>Dream Score: {points}</p>
    </div>
  </div>
);
```

### Data Visualization
```javascript
// Radar Chart Implementation
const RadarChartComponent = ({ fields }) => {
  const chartData = {
    labels: fields.map(field => field.label),
    datasets: [{
      data: fields.map(field => field.value),
      // ... configuration
    }]
  };
};
```

## Setup & Development

### Prerequisites
- Node.js >= 16.0.0
- npm >= 8.0.0

### Installation Steps
```bash
# Clone repository
git clone [repository-url]

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run linting
npm run lint
```

### Environment Configuration
```javascript
// vite.config.js
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": "/src"
    }
  }
});
```

## UI Component Guidelines

### Player Card Implementation
```jsx
<PlayerCard
  name={player.name}
  points={player.dreamPoints}
  type={player.type}
  teamIconUrl={player.team.icon}
  profileImage={player.image}
/>
```

### Weather Card Usage
```jsx
<WeatherCard
  temperature={matchData.temperature}
  humidity={matchData.humidity}
  condition={weatherCondition}
/>
```

### Team Builder Interface
- Drag and drop zones for player placement
- Real-time validation of team composition
- Position-specific player filtering
- Team balance indicators

## Performance Optimization

### Implemented Optimizations
- Code splitting using React.lazy()
- Image optimization and lazy loading
- Memoization of expensive calculations
- Efficient re-rendering using React.memo

### Caching Strategy
- Browser caching for static assets
- In-memory caching for frequently accessed data
- Local storage for user preferences

## Contributing

Please read our contributing guidelines before submitting pull requests. Follow the coding standards and commit message conventions.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and queries, please raise an issue in the repository or contact the maintainers.
