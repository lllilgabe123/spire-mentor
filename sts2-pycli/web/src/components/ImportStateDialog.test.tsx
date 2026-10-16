import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import ImportStateDialog from "./ImportStateDialog";
import { api, type ImportPreview } from "../api";

afterEach(() => vi.restoreAllMocks());

const STATE = '{"type":"advice_request","advice_for":"combat_action"}';

function mockPreview() {
  return vi.spyOn(api.live, "importPreview").mockResolvedValue({
    has_combat: true,
    surface: "combat",
    run: null,
  } as unknown as ImportPreview);
}

test("without a live game there is no live-game button", () => {
  render(<ImportStateDialog onImported={() => {}} onClose={() => {}} />);
  expect(screen.queryByRole("button", { name: /from live game/i })).toBeNull();
});

test("the live-game button fills the paste box and previews it", async () => {
  const preview = mockPreview();
  const liveGame = vi.fn().mockResolvedValue(STATE);
  render(<ImportStateDialog onImported={() => {}} onClose={() => {}} liveGame={liveGame} />);
  expect(liveGame).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole("button", { name: /from live game/i }));

  await waitFor(() => expect(screen.getByRole("textbox")).toHaveProperty("value", STATE));
  await waitFor(() => expect(preview).toHaveBeenCalledWith(STATE));
});

test("autoLive fetches once on open, not again on re-render", async () => {
  mockPreview();
  const liveGame = vi.fn().mockResolvedValue(STATE);
  const { rerender } = render(
    <ImportStateDialog onImported={() => {}} onClose={() => {}} liveGame={liveGame} autoLive />);
  await waitFor(() => expect(screen.getByRole("textbox")).toHaveProperty("value", STATE));
  rerender(<ImportStateDialog onImported={() => {}} onClose={() => {}} liveGame={liveGame} autoLive />);
  expect(liveGame).toHaveBeenCalledTimes(1);
});

test("a 404 from the game reads as 'no position yet'", async () => {
  const liveGame = vi.fn().mockRejectedValue(new Error('404 {"error":"no_state"}'));
  render(<ImportStateDialog onImported={() => {}} onClose={() => {}} liveGame={liveGame} autoLive />);
  expect(await screen.findByRole("alert")).toHaveProperty(
    "textContent", expect.stringMatching(/no position yet/));
});
