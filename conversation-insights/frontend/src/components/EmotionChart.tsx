import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';
import { useState } from 'react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface EmotionTimelineEntry {
  time: number;
  emotions: Record<string, number>;
}

interface EmotionChartProps {
  speakerATimeline: EmotionTimelineEntry[];
  speakerBTimeline: EmotionTimelineEntry[];
  speakerALabel: string;  // "CALLER" or "SPEAKER_A"
  speakerBLabel: string;  // "AGENT" or "SPEAKER_B"
  topEmotions: string[];  // Top 5 emotions to show by default
}

export default function EmotionChart({
  speakerATimeline,
  speakerBTimeline,
  speakerALabel,
  speakerBLabel,
  topEmotions
}: EmotionChartProps) {
  const [showAll, setShowAll] = useState(false);

  // Emotion color palette
  const emotionColors: Record<string, string> = {
    'Joy': '#22c55e',
    'Sadness': '#3b82f6',
    'Anger': '#ef4444',
    'Frustration': '#f97316',
    'Anxiety': '#8b5cf6',
    'Calmness': '#06b6d4',
    'Excitement': '#eab308',
    'Contentment': '#10b981',
    'Fear': '#dc2626',
    'Surprise': '#f59e0b',
    'Confusion': '#6b7280',
    'Determination': '#14b8a6'
  };

  const getColor = (emotion: string) => emotionColors[emotion] || '#6b7280';

  if (!speakerATimeline || speakerATimeline.length === 0) {
    return (
      <div className="text-purple-300 text-xs text-center py-4">
        No emotion data available
      </div>
    );
  }

  // Determine which emotions to show
  const emotionsToShow = showAll
    ? Object.keys(speakerATimeline[0]?.emotions || {})
    : topEmotions.slice(0, 5);

  // Prepare chart data
  const datasets = emotionsToShow.flatMap(emotion => [
    // Speaker A - solid line
    {
      label: `${speakerALabel} - ${emotion}`,
      data: speakerATimeline.map(entry => ({
        x: entry.time,
        y: entry.emotions[emotion] || 0
      })),
      borderColor: getColor(emotion),
      backgroundColor: getColor(emotion) + '20',
      borderWidth: 2,
      borderDash: [],  // Solid line
      pointRadius: 0,
      tension: 0.4
    },
    // Speaker B - dashed line
    {
      label: `${speakerBLabel} - ${emotion}`,
      data: speakerBTimeline.map(entry => ({
        x: entry.time,
        y: entry.emotions[emotion] || 0
      })),
      borderColor: getColor(emotion),
      backgroundColor: getColor(emotion) + '10',
      borderWidth: 2,
      borderDash: [5, 5],  // Dashed line
      pointRadius: 0,
      tension: 0.4
    }
  ]);

  const chartData = {
    datasets
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: 'linear' as const,
        title: {
          display: true,
          text: 'Time (seconds)',
          color: '#c4b5fd'
        },
        ticks: { color: '#c4b5fd' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      y: {
        title: {
          display: true,
          text: 'Emotion Intensity',
          color: '#c4b5fd'
        },
        min: 0,
        max: 1,
        ticks: { color: '#c4b5fd' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'bottom' as const,
        labels: {
          color: '#c4b5fd',
          font: { size: 10 },
          usePointStyle: true,
          padding: 8
        }
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false
      }
    }
  };

  return (
    <div className="emotion-chart-container">
      <div className="h-64 mb-4">
        <Line data={chartData} options={options} />
      </div>

      <div className="flex items-center justify-between">
        <button
          onClick={() => setShowAll(!showAll)}
          className="text-xs text-purple-300 hover:text-purple-100 underline"
        >
          {showAll ? 'Show top 5 only' : 'Show all emotions'}
        </button>

        <div className="text-xs text-purple-400">
          Solid = {speakerALabel} | Dashed = {speakerBLabel}
        </div>
      </div>
    </div>
  );
}
