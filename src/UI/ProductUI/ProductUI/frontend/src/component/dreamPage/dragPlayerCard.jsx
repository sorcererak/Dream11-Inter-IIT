import { useDrag } from "react-dnd";
import { ItemType } from "../../constants";
import PlayerCard from "../playerCard";

export default function DragPlayerCard({
  player,
  isDraggable = true,
  onAddToField,
}) {
  const [{ isDragging }, drag] = useDrag(() => ({
    type: ItemType.PLAYER,
    item: player,
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  }));

  return (
    <div
      ref={isDraggable ? drag : null}
      style={{
        backgroundColor: "transparent",
        opacity: isDragging ? 0.5 : 1,
        cursor: isDraggable ? "move" : "default",
      }}
    >
      <PlayerCard {...player} onAddToField={onAddToField} />
    </div>
  );
}
